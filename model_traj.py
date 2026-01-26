import torch.nn as nn
import torch
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0, bidirectional=False):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True  # 输入输出shape为 (batch, seq_len, feature)
        )
    
    def forward(self, x):
            """
            x: (batch, seq_len, input_dim)
            """
            output, (hn, cn) = self.lstm(x)
            # output: (batch, seq_len, hidden_dim * num_directions)

            return output

# MOE共享专家和独立专家
class Expert(nn.Module):
    def __init__(self, input_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim*2)
        self.fc2 = nn.Linear(input_dim*2, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class MOEModel(nn.Module):
    def __init__(self, top_k, num_shared_experts, num_independent_experts, input_dim, vision_dim):
        super(MOEModel, self).__init__()
        self.num_independent_experts = num_independent_experts
        self.topk = top_k
        # Shared experts
        self.shared_experts = nn.ModuleList([Expert(input_dim) for _ in range(num_shared_experts)])
        # Independent experts
        self.independent_experts = nn.ModuleList([Expert(input_dim) for _ in range(num_independent_experts)])
        # Gating network for independent experts
        self.traj_gating_network = nn.Linear(input_dim, num_independent_experts)
        self.vision_induce_gating = nn.Linear(vision_dim, num_independent_experts)

    def forward(self, feature_vectors, object_type, vision_inter):
        '''
        feature_vectors: (batch_size, num_node, his_len, input_dim)
        object_type: (batch_size, num_node, his_len)
        vision_inter: (batch_size, num_node, vision_dim)
        '''
        batch_size, num_nodes, his_len, input_dim = feature_vectors.size()

        # Reshape input to (batch_size * num_nodes * his_len, input_dim)
        x_flat = feature_vectors.reshape(batch_size * num_nodes * his_len, input_dim)

        # Get independent experts gating scores and Add vision_inter
        vision_induce_score = self.vision_induce_gating(vision_inter).unsqueeze(2).expand(-1, -1, his_len, -1)
        vision_induce_score = vision_induce_score.reshape(-1, self.num_independent_experts)
        gating_scores = self.traj_gating_network(x_flat) + vision_induce_score  # (batch_size * num_nodes * his_len, num_independent_experts)
        # select top-k independent experts
        gating_scores = F.softmax(gating_scores, dim=1)
        indices = torch.topk(gating_scores, self.topk, dim=-1)[1]
        weights = gating_scores.gather(1, indices).type_as(feature_vectors)
        weights /= weights.sum(dim=-1, keepdim=True)
        
        # Select shared expert based on object type
        mapped = torch.zeros_like(object_type, requires_grad=False).to(feature_vectors.device)
        mapped[object_type == 0] = 0
        mapped[(object_type == 1) | (object_type == 2)] = 1
        mapped[(object_type >= 3) & (object_type <= 6)] = 2
        object_types = F.one_hot(mapped.long().clamp(min=0), num_classes=3).unsqueeze(-1).float()
        object_types[mapped == -1] = 0.0
        object_types = object_types.view(-1,3,1)  # (batch_size * his_len,3,1)
          
        # shared_expert_output (batch_size * num_nodes * his_len, 3, output_dim)
        shared_expert_outputs = torch.stack([expert(x_flat) for expert in self.shared_experts], dim=1)
        # selected_shared_expert_output (batch_size * num_nodes * his_len, output_dim)
        shared_expert_output = torch.mul(shared_expert_outputs, object_types).sum(dim=1)
        
        # Get independent expert outputs 
        independent_expert_output = torch.zeros_like(shared_expert_output)
        counts = torch.bincount(indices.flatten(), minlength=self.num_independent_experts)
        for i in range(self.num_independent_experts):
            if counts[i] == 0:
                continue
            expert = self.independent_experts[i]
            idx, top = torch.where(indices == i)
            independent_expert_output[idx] += expert(x_flat[idx]) * (weights[idx, top].unsqueeze(1))
        # Reshape output to (batch_size, num_nodes, his_len, output_dim)
        output = 0.5 * shared_expert_output + 0.5 * independent_expert_output
        output = output.view(batch_size, num_nodes, his_len, -1)
        return output
    
class RelationAwareGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, num_relation):
        super(RelationAwareGATLayer, self).__init__()
        self.gat_enc = nn.Linear(in_features, out_features)
        self.num_heads = num_heads
        self.num_relation = num_relation
        self.head_dim = int(out_features // num_heads)
        self.attn = nn.Linear(2 * self.head_dim, 1)
        self.rela_w = nn.Linear(self.num_relation, out_features)
    
    
    def forward(self, h, adj, relations):
        """
        h: (batch_size, node_num, seq_len, encoder_dim)
        adj: (batch_size, node_num, seq_len)
        relations: (batch_size, seq_len, node_num-1, 4)
        """
        batch_size, node_num, seq_len, _ = h.size()
        h = h.permute(0,2,1,3)
        adj = adj.permute(0,2,1)
        
        self_dist = torch.full((batch_size, seq_len, 1), float('inf')).to(h.device)
        neigh_dist = relations[..., 0].masked_fill((1 - adj[:,:,1:]).bool(), float('inf'))
        dist = torch.cat((self_dist, neigh_dist), dim=-1)
        score = torch.exp(-dist / (node_num-1)**2) # (batch_size, seq_len, node_num)
        
        # GAT
        enc = self.gat_enc(h) # (batch_size,seq_len,node_num,hid_dim)
        target = enc[:,:,0,:].view(batch_size, seq_len, self.num_heads, self.head_dim).unsqueeze(3)
        target = target.repeat(1, 1, 1, node_num, 1)  # (batch_size, seq_len, num_heads, node_num, head_dim)
        rela_emned = self.rela_w(relations)
        h0_neighbor = torch.concat([enc[:,:,0,:].unsqueeze(2), enc[:,:,1:] + rela_emned], dim=2)
        h0_neighbor = h0_neighbor.view(batch_size, seq_len, node_num, self.num_heads, self.head_dim)
        h0_neighbor = h0_neighbor.permute(0,1,3,2,4) # (batch_size, seq_len, head_num, node_num, head_dim)
        # Compute attention scores 
        energy = torch.cat([target, h0_neighbor], dim=-1)  # (batch_size, seq_len, num_heads, node_num, 2*head_dim)
        energy = self.attn(energy).squeeze(-1)
        attention = F.softmax(energy, dim=-1) # (batch_size, seq_len, num_heads, node_num)
        attention = F.softmax(attention * score.unsqueeze(2), dim=-1)

        # Weighted sum of neighbors
        h_agg = torch.matmul(attention.unsqueeze(3), h0_neighbor).squeeze()  # (batch_size, seq_len, num_heads, head_dim)
        h_agg = h_agg.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, out_features)
        # add self
        h_agg = h_agg + enc[:, :, 0, :]
        return h_agg

class Trajectory_Modal_Learner(nn.Module):
    def __init__(self, args):
        super(Trajectory_Modal_Learner, self).__init__()
        self.args = args
        self.locat_embed = nn.Linear(2, self.args.encoder_dim)
        self.grid_embed = nn.Linear(2, self.args.encoder_dim)
        self.dynamic_embed = nn.Linear(5, self.args.encoder_dim)
        self.type_embed = nn.Embedding(num_embeddings=args.road_user_types+1, padding_idx=0, embedding_dim=self.args.encoder_dim)
        self.vision_pool = nn.AvgPool2d(kernel_size=args.vtm_patch_size, stride=args.vtm_patch_size)
        self.vision_to_enc = nn.Linear(args.vit_hidden_size, self.args.encoder_dim)
        self.LSTM = LSTM(args.encoder_dim, args.encoder_dim)
        self.MOE = MOEModel(args.top_k, args.num_shared_experts, args.num_independent_experts, args.encoder_dim, args.local_patch_num //4)
        self.GAT = RelationAwareGATLayer(args.encoder_dim, args.gat_hid_dim, args.gat_nhead, args.num_relations)
        self.gat_enc = nn.Linear(args.encoder_dim, args.gat_hid_dim)
        if args.train_model:
            self.random_initialize()
    
    def random_initialize(self):
        for param in self.parameters():
            if param.requires_grad:
                if len(param.shape) < 2:
                    torch.nn.init.xavier_uniform_(param.unsqueeze(0))
                else:
                    torch.nn.init.xavier_uniform_(param)

    def map_patch_idx(self, grid_xid, grid_yid, patch_num_width):
        patch_grid_num = self.args.vtm_patch_size // self.args.grid_size
        patch_xid, patch_yid = grid_xid//patch_grid_num, grid_yid//patch_grid_num
        patch_idx = patch_yid + patch_xid * patch_num_width + 1
        return patch_idx
        
    def forward(self, x_data, x_neighbor_dis, x_mask, vision_inter):
        '''
        x_data.shape (batch_size, node_num, inp_seq_len, feat_num)
        x_neighbor_dis.shape (batch_size, inp_seq_len, node_num-1, 4)
        x_mask.shape (batch_size, node_num, inp_seq_len)
        vision_inter.shape (batch_size, node_num, width, length)
        '''
        batch_size, num_node, inp_his_len, _ = x_data.size()
        # trajectory encoding
        his_target_location = x_data[:,0,:,:2] - x_data[:,0,0,:2].unsqueeze(1)
        his_target_grid = x_data[:,0,:,2:4]
        his_target_grid[:, :, 0] = his_target_grid[:, :, 0] / self.args.x_grid_num
        his_target_grid[:, :, 1] = his_target_grid[:, :, 1] / self.args.y_grid_num
        his_target_dynamic = x_data[:,0,:,4:-1]
        target_type = x_data[:,0,:,-1]
        target_locat_embedding = self.locat_embed(his_target_location)
        target_grid_embedding = self.grid_embed(his_target_grid)
        target_dynamic_embedding = self.dynamic_embed(his_target_dynamic)
        target_type_embedding = self.type_embed(target_type.long())
        target_embedding = target_grid_embedding + target_dynamic_embedding + target_type_embedding
        
        his_neighbor_location = x_data[:,1:,:,:2] - x_data[:,1:,0,:2].unsqueeze(2)
        his_neighbor_grid = x_data[:,1:,:,2:4]
        his_neighbor_grid[:, :, 0] = his_neighbor_grid[:, :, 0] / self.args.x_grid_num
        his_neighbor_grid[:, :, 1] = his_neighbor_grid[:, :, 1] / self.args.y_grid_num
        his_neighbor_dynamic = x_data[:,1:,:,4:-1]
        his_neighbor_type = x_data[:,1:,:,-1]
        his_neighbor_type[his_neighbor_type == -1] = 0
        neighbor_locat_embedding = self.locat_embed(his_neighbor_location)
        neighbor_grid_embedding = self.grid_embed(his_neighbor_grid)
        neighbor_dynamic_embedding = self.dynamic_embed(his_neighbor_dynamic)
        neighbor_type_embedding = self.type_embed(his_neighbor_type.long())
        neighbor_embedding = neighbor_grid_embedding + neighbor_dynamic_embedding + neighbor_type_embedding
  
        x = torch.concat([target_embedding.unsqueeze(1), neighbor_embedding], dim = 1)

        # fuse map semactic
        # grid_x, grid_y = x_data[:,:,:,2], x_data[:,:,:,3]
        # patch_idx = self.map_patch_idx(grid_x, grid_y, patch_num_width)
        # patch_idx = patch_idx * x_mask
        # flat_idx = patch_idx.view(-1).long()
        # flat_embeddings = map_embed[0][flat_idx]
        # map_grid_embed = flat_embeddings.view(batch_size, num_node, inp_his_len, self.args.vit_hidden_size)
        # map_grid_embed = self.vision_to_enc(map_grid_embed)
        # x = x + map_grid_embed

        # lstm
        x = self.LSTM(x.reshape(batch_size * num_node, inp_his_len, -1))
        x = x.reshape(batch_size, num_node, inp_his_len, -1)

        # Semantic-aware MoE
        vision_inter = self.vision_pool(vision_inter).view(batch_size, num_node, -1)
        moe_out = self.MOE(x, x_data[...,-1], vision_inter) # (batch,node,seq_len,enc_dim)
        
        # Relation-aware GAT
        gat_out = self.GAT(moe_out, x_mask, x_neighbor_dis) # (batch, seq_len,gat_hid_dim)


        residual = gat_out + self.gat_enc(target_locat_embedding)

        return residual
    



    
