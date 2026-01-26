import torch
from torch import nn
import math
import torch.nn.functional as F
from transformers import ViTModel
from model_vision import Vision_Modal_Learner
from model_traj import Trajectory_Modal_Learner

class SampleCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.vtm_hidden_size % config.vtm_num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.vtm_hidden_size,} is not a multiple of the number of attention "
                f"heads {config.vtm_num_attention_heads}."
            )

        self.num_attention_heads = config.vtm_num_attention_heads
        self.attention_head_size = int(config.vtm_hidden_size / config.vtm_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.vtm_hidden_size, self.all_head_size, bias=config.vtm_qkv_bias)
        self.key = nn.Linear(config.vtm_hidden_size, self.all_head_size, bias=config.vtm_qkv_bias)
        self.value = nn.Linear(config.vtm_hidden_size, self.all_head_size, bias=config.vtm_qkv_bias)

        self.dropout = nn.Dropout(config.vtm_attention_dropout_prob)
        self.layernorm_before = nn.LayerNorm(config.vtm_hidden_size, eps=config.vtm_layer_norm_eps)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_embed, vision_embed, attention_mask=None, head_mask=None, output_attentions=False):
        vision_embed = self.layernorm_before(vision_embed)
        
        mixed_query_layer = self.query(query_embed)

        key_layer = self.transpose_for_scores(self.key(vision_embed))
        value_layer = self.transpose_for_scores(self.value(vision_embed))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class TrajSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.vtm_hidden_size % config.vtm_num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.vtm_hidden_size,} is not a multiple of the number of attention "
                f"heads {config.vtm_num_attention_heads}."
            )

        self.num_attention_heads = config.vtm_num_attention_heads
        self.attention_head_size = int(config.vtm_hidden_size / config.vtm_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.vtm_hidden_size, self.all_head_size, bias=config.vtm_qkv_bias)
        self.key = nn.Linear(config.vtm_hidden_size, self.all_head_size, bias=config.vtm_qkv_bias)
        self.value = nn.Linear(config.vtm_hidden_size, self.all_head_size, bias=config.vtm_qkv_bias)

        self.dropout = nn.Dropout(config.vtm_attention_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class Fusion_SelfOutput(nn.Module):
    """
    The residual connection is defined in Fusion_Layer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.vtm_hidden_size, config.vtm_hidden_size)
        self.dropout = nn.Dropout(config.vtm_hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
    
class Modal_Self_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TrajSelfAttention(config)
        self.output = Fusion_SelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class Fusion_Intermediate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.vtm_hidden_size, config.vtm_intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

class Fusion_Output(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.vtm_intermediate_size, config.vtm_hidden_size)
        self.dropout = nn.Dropout(config.vtm_hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class Fusion_Layer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.self_attention = Modal_Self_Attention(config)
        self.intermediate = Fusion_Intermediate(config)
        self.output = Fusion_Output(config)
        self.layernorm_before = nn.LayerNorm(config.vtm_hidden_size, eps=config.vtm_layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.vtm_hidden_size, eps=config.vtm_layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):

        self_attention_outputs = self.self_attention(
            self.layernorm_before(hidden_states),
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        output = self_attention_outputs[0]
        if output_attentions:
            atten_score = self_attention_outputs[1]
        
        hidden_states = output + hidden_states.to(output.device)
        layer_output = self.layernorm_after(hidden_states)
        
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        # layer_output = self.layernorm_after(layer_output)

        outputs = (layer_output,) if not output_attentions else (layer_output, atten_score)

        return outputs

class Modal_Embeddings(nn.Module):
    """
    Construct the traj and patch embeddings.

    traj embeddings are equivalent to BERT embeddings.

    Patch embeddings are equivalent to ViT embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.BOI_token = nn.Parameter(torch.zeros(1, 1, config.vtm_hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vtm_hidden_size))
        self.traj_position_embeddings = nn.Embedding(config.his_len + 1, config.vtm_hidden_size)
        self.patch_position_embeddings = nn.Parameter(torch.zeros(1, config.sample_query_len + 1, config.vtm_hidden_size)) # If position_embed is needed?
        # modality type (traj/patch) embeddings
        self.token_type_embeddings = nn.Embedding(config.vtm_modality_type_vocab_size, config.vtm_hidden_size)
        self.dropout = nn.Dropout(config.vtm_hidden_dropout_prob)
        self.config = config
    
    def forward(self, traj_embeds, image_embeds, traj_token_type_idx = 0, image_token_type_idx = 1):
        
        # add position embed and BOI token
        image_embeds = torch.cat([self.BOI_token.expand(image_embeds.size(0), -1, -1), image_embeds], dim=1)
        image_embeds = image_embeds + self.patch_position_embeddings.expand(image_embeds.size(0), -1, -1)
        
        # add position embed and cls token
        position_ids = torch.arange(traj_embeds.size(1) + 1, dtype = torch.long, device = traj_embeds.device).unsqueeze(0)
        traj_embeds = torch.cat([self.cls_token.expand(traj_embeds.size(0), -1, -1), traj_embeds], dim=1)
        traj_embeds = traj_embeds + self.traj_position_embeddings(position_ids)
        
        traj_embeds = traj_embeds + self.token_type_embeddings(
            torch.ones((traj_embeds.size(0), traj_embeds.size(1)), dtype=torch.long, device=traj_embeds.device)
        )
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.zeros((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=image_embeds.device)
        )

        # concatenate
        embeddings = torch.cat([traj_embeds, image_embeds], dim=1)

        return embeddings

class MultiModal_Fusion_Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Fusion_Layer(config) for _ in range(config.vtm_num_hidden_layers)])
        
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        all_hidden_states, all_attentions = (), ()

        for i, layer_module in enumerate(self.layer):
           
            all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions=True)

            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        all_hidden_states = all_hidden_states + (hidden_states,)
        
        return all_hidden_states[-1], all_attentions
    
class TrajDecoder(nn.Module):

    def __init__(self, args) -> None:
        super(TrajDecoder, self).__init__()
        min_scale: float = 1e-3
        self.args = args
        self.input_size = self.args.vtm_hidden_size
        self.hidden_size = args.traj_dec_hidden_size
        self.pred_len = args.pred_len
        self.num_modes = args.traj_modal_num
        self.min_scale = min_scale
        self.args = args
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size))
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 1))
        self.multihead_proj_global = nn.Sequential(
                                    nn.Linear(self.input_size , self.num_modes * self.hidden_size),
                                    nn.LayerNorm(self.num_modes * self.hidden_size))
        self.multihead_proj_hid = nn.Sequential(
                                    nn.Linear(self.input_size , self.hidden_size),
                                    nn.LayerNorm(self.hidden_size))
        if args.train_model:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for ih in param.chunk(4, 0):
                        nn.init.xavier_uniform_(ih)
                elif 'weight_hh' in name:
                    for hh in param.chunk(4, 0):
                        nn.init.orthogonal_(hh)
                elif 'weight_hr' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias_ih' in name:
                    nn.init.zeros_(param)
                elif 'bias_hh' in name:
                    nn.init.zeros_(param)
                    nn.init.ones_(param.chunk(4, 0)[1])

    def forward(self, global_embed: torch.Tensor, hidden_state):
        global_embed = self.multihead_proj_global(global_embed).view(-1, self.pred_len, self.num_modes, self.hidden_size)
        global_embed = global_embed.permute(1, 2, 0, 3).contiguous()
        global_embed = global_embed.view(self.pred_len, -1, self.hidden_size) # [H, F x N, D]
        hidden_state = self.multihead_proj_hid(hidden_state).unsqueeze(0)  # [1, N, D]
        local_embed = hidden_state.repeat(self.num_modes, 1, 1)  # [F, N, D]
        pi = self.pi(local_embed).squeeze(-1).t()  # [N, F]
        local_embed = local_embed.reshape(-1, self.hidden_size).unsqueeze(0).repeat(self.pred_len, 1, 1)  # [H, F x N, D]
        out = self.mlp(torch.cat([global_embed, local_embed], dim=-1))
        out = out.transpose(0, 1)  # [F x N, H, D]
        loc = self.loc(out)  # [F x N, H, 2]
        loc = loc.view(self.num_modes, -1, self.pred_len, 2) # [F, N, H, 2]
        return loc, pi # [F, N, H, 2], [N, F]
    
class KD_loss(nn.Module):
    def __init__(self, alpha, temperature):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, z_patch, z_pretrain):
        z_patch = z_patch.reshape(-1, z_patch.size(-1))
        z_pretrain = z_pretrain.reshape(-1, z_pretrain.size(-1))

        mse_loss = F.mse_loss(z_patch, z_pretrain)

        p_patch = F.log_softmax(z_patch / self.temperature, dim=-1)
        p_pretrain = F.softmax(z_pretrain / self.temperature, dim=-1)
        kl_loss = F.kl_div(p_patch, p_pretrain, reduction="batchmean") * (self.temperature ** 2)

        loss = self.alpha * mse_loss + (1 - self.alpha) * kl_loss
        return loss
    
class SoftTargetCrossEntropyLoss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        xy_diff = pred - target
        mse_loss = torch.mean(torch.mean(torch.sqrt(torch.sum(xy_diff ** 2, dim=2)), dim=1))
        return mse_loss
    
class Vision_Trajectory_Model(nn.Module):
    def __init__(self, args):
        super(Vision_Trajectory_Model, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.patch_size = args.vtm_patch_size
        self.in_channels = args.in_channels
        self.image_size = args.local_image_size
        self.sample_query_len = args.sample_query_len
        self.embeddings = Modal_Embeddings(args)
        self.vision_modal_learner = Vision_Modal_Learner(args)
        self.trajectory_modal_learner = Trajectory_Modal_Learner(args)
        self.fusion_network = MultiModal_Fusion_Network(args)
        self.traj_to_input = nn.Linear(args.gat_hid_dim, args.vtm_hidden_size)
        self.traj_pred = TrajDecoder(args)
        self.vit_path = args.pretrain_vit_path
        self.vit_model = ViTModel.from_pretrained(self.vit_path).to(self.args.device)
        self.sample_model = SampleCrossAttention(args)
        self.image_kd_loss = KD_loss(alpha=0.5, temperature=0.5)
        self.layernorm = nn.LayerNorm(args.vtm_hidden_size, args.vtm_layer_norm_eps)
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

    def mdn_loss(self, y, y_prime):
        batch_size=y.shape[0]  #[N, H, 2]
        # [F, N, H, 2], [N, F]
        out_mu, out_pi = y_prime 
        y_hat = out_mu
        reg_loss, cls_loss = 0, 0
        full_pre_tra = []
        l2_norm = (torch.norm(out_mu - y, p=2, dim=-1)).mean(dim=-1)   # [F, N]
        best_ade_mode = l2_norm.argmin(dim=0)
        last_norm = (torch.norm(out_mu[:,:,-1] - y[:,-1], p=2, dim=-1))   # [F, N]
        best_fde_mode = last_norm.argmin(dim=0)
        softmax_pi = F.softmax(out_pi, dim=-1)
        soft_target = F.softmax(-last_norm / self.args.pred_len, dim=0).t().detach() # [N, F]
        cls_loss += self.cls_loss(softmax_pi, soft_target)
        best_pi_mode = softmax_pi.argmax(dim=-1)
        y_hat_best = y_hat[best_pi_mode, torch.arange(batch_size)]
        reg_loss += self.reg_loss(y_hat_best, y)
        # cls_loss = F.kl_div(F.log_softmax(out_pi, dim=-1), soft_target, reduction='batchmean')
        loss = reg_loss + cls_loss
        #best ADE
        sample_k = out_mu[best_pi_mode, torch.arange(batch_size)] #[N, H, 2]
        full_pre_tra.append(sample_k)
        softmax_pi = F.softmax(out_pi, dim=-1)
        best_pi = softmax_pi[torch.arange(batch_size), best_pi_mode]
        # best FDE
        full_pre_tra.append(sample_k)
        return loss, full_pre_tra, best_pi
    
    def image_scale(self, img_tensor):
        mean = img_tensor.mean(dim=[0,2,3]).view(1,3,1,1)
        std = img_tensor.std(dim=[0,2,3]).view(1,3,1,1)
        img_norm = (img_tensor - mean) / (std + 1e-6)
        return img_norm

    def forward(self, x_data, x_neighbor_dis, x_mask, target, map_image, trajectory_images, local_patch_xids, local_patch_yids, target_label_iamge):

        # get vision Modal
        vision_node_embed, vision_inter = self.vision_modal_learner(map_image, trajectory_images, local_patch_xids, local_patch_yids)
        # image patch embedding KD-loss
        label_image = target_label_iamge.permute(0, 3, 1, 2)
        with torch.no_grad():
            target_vision_embeddings = self.vit_model(label_image, interpolate_pos_encoding = True).last_hidden_state
        image_loss = self.image_kd_loss(vision_inter, target_vision_embeddings[:,1:,:])
        # sample
        queries = torch.randn(self.batch_size, self.sample_query_len, self.args.vtm_hidden_size).to(self.args.device)
        vision_logits = self.sample_model(queries, vision_inter)[0]
        # get trajectory Modal
        trajectory_inter = self.trajectory_modal_learner(x_data, x_neighbor_dis, x_mask, vision_node_embed)
        # fusion network
        trajectory_inter = self.traj_to_input(trajectory_inter)
        embeddings = self.embeddings(trajectory_inter, vision_logits)
        fusion_output, attentions = self.fusion_network(embeddings, output_attentions = False)
        if attentions:
            attentions = torch.stack(attentions, dim=0)
            attentions = attentions.permute(1,0,2,3,4)
        # traj pred
        cls_embed = fusion_output[:, 0, :]
        traj_fusion_embed = fusion_output[:, 1:1+trajectory_inter.size(1):, :]
        loc, pi = self.traj_pred(traj_fusion_embed, cls_embed)
        
        label = target[:,:,:2] - x_data[:,0,-1,:2].unsqueeze(1) # 相对历史最后一个轨迹点的距离
        
        traj_loss, full_pre_tra, best_pi = self.mdn_loss(label, (loc, pi))

        # loss
        loss = image_loss + traj_loss
        return loss, full_pre_tra, best_pi

    def grad_state_dict(self):
        # 筛选出 requires_grad=True 的参数
        params_to_save = filter(lambda p: p[1].requires_grad, self.named_parameters())
        save_list = [p[0] for p in params_to_save]
        return  {name: param.detach() for name, param in self.state_dict().items() if name in save_list}
        
    
    def save(self, path:str):
        
        selected_state_dict = self.grad_state_dict()
        torch.save(selected_state_dict, path)
    
    def load(self, path:str):

        loaded_params = torch.load(path, map_location=f"cuda:{self.args.gpu}")
        self.load_state_dict(loaded_params,strict=False)
    
    def params_num(self):
        total_params = sum(p.numel() for p in self.parameters()) # torch.numel()返回张量的元素个数
        total_params += sum(p.numel() for p in self.buffers())
        
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, total_trainable_params