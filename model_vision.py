import torch
import torch.nn as nn
import torch.nn.functional as F
from Mask_ViT import ViTModel
# from transformers import ViTModel
# from peft import get_peft_model, LoraConfig
from einops import rearrange

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels, scale_patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.scale_patch_size = scale_patch_size

        self.norm = self.Normalize(in_channels)
        self.q = nn.Sequential(
            nn.Conv2d(
            in_channels, in_channels * self.scale_patch_size, kernel_size=self.scale_patch_size, stride=self.scale_patch_size, padding=0),
            nn.BatchNorm2d(in_channels * self.scale_patch_size),
            nn.Tanh()
        )
        self.k = nn.Sequential(
            nn.Conv2d(
            in_channels, in_channels * self.scale_patch_size, kernel_size=self.scale_patch_size, stride=self.scale_patch_size, padding=0),
            nn.BatchNorm2d(in_channels * self.scale_patch_size),
            nn.Tanh()
        )
        self.v = nn.Sequential(
            nn.Conv2d(
            in_channels, in_channels * self.scale_patch_size, kernel_size=self.scale_patch_size, stride=self.scale_patch_size, padding=0),
            nn.BatchNorm2d(in_channels * self.scale_patch_size),
            nn.Tanh()
        )
        self.proj_out = nn.Sequential(
            nn.ConvTranspose2d(
            in_channels * self.scale_patch_size, in_channels, kernel_size=self.scale_patch_size, stride=self.scale_patch_size, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def Normalize(self, in_channels):
        return torch.nn.BatchNorm2d(in_channels)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_
    
class SpatialCrossAttention(nn.Module):
    def __init__(self, in_channels, node_num, scale_patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.scale_patch_size = scale_patch_size
        self.node_num = node_num
        self.q = nn.Sequential(
            nn.Conv2d(
            in_channels, in_channels * self.scale_patch_size, kernel_size=self.scale_patch_size, stride=self.scale_patch_size, padding=0),
            nn.BatchNorm2d(in_channels * self.scale_patch_size),
            nn.Tanh()
        )
        self.k = nn.Sequential(
            nn.Conv2d(
            in_channels, in_channels * self.scale_patch_size, kernel_size=self.scale_patch_size, stride=self.scale_patch_size, padding=0),
            nn.BatchNorm2d(in_channels * self.scale_patch_size),
            nn.Tanh()
        )
        self.v = nn.Sequential(
            nn.Conv2d(
            in_channels, in_channels * self.scale_patch_size, kernel_size=self.scale_patch_size, stride=self.scale_patch_size, padding=0),
            nn.BatchNorm2d(in_channels * self.scale_patch_size),
            nn.Tanh()
        )
        self.proj_out = nn.Sequential(
            nn.ConvTranspose2d(
            in_channels * self.scale_patch_size, in_channels, kernel_size=self.scale_patch_size, stride=self.scale_patch_size, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        h_ = x
        q_h_ = h_[:,:1,:,:]
        kv_h_ = h_[:,1:,:,:].permute(0,2,3,1).contiguous()
        kv_h_ = kv_h_.view(kv_h_.shape[0], kv_h_.shape[1], -1).unsqueeze(1)
        q = self.q(q_h_)
        k = self.k(kv_h_)
        v = self.v(kv_h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_).squeeze()

        return h_ + x[:,0,:,:]
    
class ChannelInteractionModel(nn.Module):
    def __init__(self, channels, image_num, scale_patch_size):
        super().__init__()
        self.image_num = image_num
        self.channels = channels
        
        # spatial self-attention
        self.spatial_attention = SpatialSelfAttention(channels, scale_patch_size)
        
        # cross different channel in same image
        self.image_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)

        # cross same channel in different image
        self.cross_agent_attention = SpatialCrossAttention(1, image_num, scale_patch_size)

    def forward(self, x, map_semantic):
        """
        x: (B, image_num, C, H, W)
        map_semantic: (B  * image_num, C, H, W)
        """
        B, image_num, C, H, W = x.shape
        map_semantic = map_semantic.view(B, image_num, C, H, W)
        map_fused = (map_semantic + x) / 2
        # spatial self-attention
        local_x = map_fused.view(B * image_num, C, H, W)
        local_x_spatial = self.spatial_attention(local_x)
        local_x_spatial = local_x_spatial.view(B, image_num, C, H, W)

        # cross different channel in same image
        inter1 = local_x_spatial.view(B * image_num, C, H, W)
        image_out = self.image_conv(inter1)   # (B * image_num, 1, H, W)
        image_out = image_out.squeeze().view(B, image_num, H, W)

        # cross same channel in different image
        inter2 = local_x_spatial.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, image_num, H, W)
        inter2 = inter2.view(B * C, image_num, H, W)
        cross_same_out = self.cross_agent_attention(inter2)  # (B * C, H, W)
        cross_same_out = cross_same_out.view(B, C, H, W)

        return image_out, cross_same_out

class ConvNet(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvNet(in_ch, out_ch, k=3, s=1, p=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x1, x2):
        identity = x1
        out = self.conv1(x2)
        out = self.norm2(self.conv2(out))
        out = out + identity
        return self.act(out)
    
class DownStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpStage(nn.Module):
    def __init__(self, image_num):
        super().__init__()
        self.residual_block = ResidualBlock(image_num, image_num)
        self.norm = nn.BatchNorm2d(image_num)

    def forward(self, last_inter, current_inter):
        last_inter = F.interpolate(last_inter, size=current_inter.shape[-2:], mode="bilinear", align_corners=False)
        inter = self.residual_block(current_inter, last_inter + current_inter)
        inter = self.norm(inter)
        return inter
    
class Map_UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale=True):
        super(Map_UpBlock, self).__init__()
        self.scale = scale
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        if self.scale:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1):
        x2 = self.conv1x1(x1)
        if self.scale:
            x2 = self.up(x2)
        return x2
    
class Map_DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout = 0.0):
        super(Map_DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x1):
        x2 = self.block(x1)
        return x2

class MultiScaleChannelAttention(nn.Module):
    def __init__(self, H1, H2, W1, W2):
        super().__init__()
        self.H1, self.H2, self.W1, self.W2 = H1, H2, W1, W2
        self.H_linear = nn.Linear(self.H1, self.H2)
        self.W_linear = nn.Linear(self.W1, self.W2)

    def forward(self, x1, x2):
        """
        x1: (B, C1, H1, W1)
        x2: (B, C2, H1, W1)
        """
        B, C1, H1, W1 = x1.shape
        B, C2, H2, W2 = x2.shape
        # Height-wise pooling
        h_pool1 = F.adaptive_avg_pool2d(x1, (H1, 1)).view(B, C1, H1)
        h_pool1 = self.H_linear(h_pool1)
        h_pool2 = F.adaptive_avg_pool2d(x2, (H2, 1)).view(B, C2, H2)
        # Width-wise pooling
        w_pool1 = F.adaptive_avg_pool2d(x1, (1, W1)).view(B, C1, W1)
        w_pool1 = self.W_linear(w_pool1)
        w_pool2 = F.adaptive_avg_pool2d(x2, (1, W2)).view(B, C2, W2)
        # Cross-scale channel interaction
        cross_h_score = torch.matmul(h_pool1, h_pool2.permute(0, 2, 1))
        cross_w_score = torch.matmul(w_pool1, w_pool2.permute(0, 2, 1))
        cross_score = cross_h_score * cross_w_score # (B, C1, C2)
        cross_score = nn.functional.softmax(cross_score, dim=-1)
        return cross_score

class Channel_Interaction_Learner(nn.Module):
    def __init__(self, base_channels, num_stages, image_num, width, height, hidden_size):
        super().__init__()
        self.num_stages = num_stages

        # image downsample
        enc_channels = [6, 12, 24, 48]
        
        self.downs = nn.ModuleList()
        self.map_downs = nn.ModuleList()
        in_ch = base_channels
        for out_ch in enc_channels:
            self.downs.append(DownStage(in_ch, out_ch))
            self.map_downs.append(Map_DownBlock(in_ch, out_ch))
            in_ch = out_ch
        
        # map-semantic upsample
        # self.map_ups = nn.ModuleList()
        # self.map_ups.append(Map_UpBlock(hidden_size, enc_channels[-1], scale=False))
        # for i in range(num_stages-1, 0, -1):
        #     self.map_ups.append(Map_UpBlock(enc_channels[i], enc_channels[i-1]))

        # vision channel interaction
        self.channel_inter = nn.ModuleList()
        for i in range(num_stages):
            self.channel_inter.append(ChannelInteractionModel(enc_channels[i], image_num, 2 ** (num_stages-i-1)))

        # image channel interaction upsample
        self.ups = nn.ModuleList()
        for i in range(num_stages - 1):
            self.ups.append(UpStage(image_num=image_num))

        # multi-scale channel Attention
        self.multi_scale_channel_atten = nn.ModuleList()
        for i in range(1, num_stages):
            h1, w1 = height // (2 ** i), width // (2 ** i)
            h2, w2 = height // (2 ** (i+1)), width // (2 ** (i+1))
            self.multi_scale_channel_atten.append(MultiScaleChannelAttention(h1, h2, w1, w2))

    def forward(self, x, map_semantic):
        '''
        map_semantic: (B * image_num, hidden_size, patch_num, patch_num)
        '''
        B, image_num, C0, H0, W0 = x.shape

        # map-semantic upsample
        # map_semantic_scale = []
        # for scale in range(self.num_stages):
        #     map_semantic = self.map_downs[scale](map_semantic)
        #     map_semantic_scale.append(map_semantic)
        
        channel_inter1, channel_inter2 = [], []
        # Every scale channel interaction
    
        for scale in range(self.num_stages):
            C, H, W  = x.shape[2:]
            x = self.downs[scale](x.view(B * image_num, C, H, W))
            x = x.view(B, image_num, x.shape[1], x.shape[2], x.shape[3])
            map_semantic = self.map_downs[scale](map_semantic)
            image_out, cross_same_out = self.channel_inter[scale](x, map_semantic)
            channel_inter1.append(image_out)
            channel_inter2.append(cross_same_out)

        # channel inter1 - upsample
        last_image_out = channel_inter1[-1]
        for scale in range(self.num_stages - 1):
            current_image_out = channel_inter1[-(scale+2)]
            last_image_out = self.ups[scale](last_image_out, current_image_out)

        # channel inter2 - cross scale channel attention
        cross_scale_score = []
        for scale in range(self.num_stages - 1):
            scale_inter1, scale_inter2 = channel_inter2[scale], channel_inter2[scale + 1]
            cross_scale_score.append(self.multi_scale_channel_atten[scale](scale_inter1, scale_inter2))
        
        for scale in range(self.num_stages - 1, 0, -1):
            scale_inter1, scale_inter2 = channel_inter2[scale-1], channel_inter2[scale]
            scale1_shape = scale_inter1.shape
            scale_inter2 = F.interpolate(scale_inter2, size=scale_inter1.shape[-2:], mode="bilinear", align_corners=False)
            scale_inter1 = scale_inter1.view(scale_inter1.shape[0], scale_inter1.shape[1], -1)
            scale_inter2 = scale_inter2.view(scale_inter2.shape[0], scale_inter2.shape[1], -1)
            new_scale_inter1 = torch.matmul(cross_scale_score[scale-1], scale_inter2)
            new_scale_inter1 = new_scale_inter1.view(scale1_shape)
            channel_inter2[scale-1] = new_scale_inter1
        
        return last_image_out, channel_inter2[0]
    
class Vision_Modal_Learner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.patch_size = args.vit_patch_size
        self.in_channels = args.in_channels
        self.local_image_size = args.local_image_size
        self.patch_embed_size = args.patch_embed_size
        self.all_patch_num = args.local_patch_num
        self.mask_vit = ViTModel(args)
        # pre_vit_path = '/data/ZhangXinyue/Multi-Model/VLM_Model/vit-base-patch16-224'
        # self.mask_vit = ViTModel.from_pretrained(pre_vit_path).to(args.device)
        # self.mask_vit.encoder.layer = self.mask_vit.encoder.layer[:args.vit_num_hidden_layers]
        self.image_enc = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)
        self.channel_inter_learner = Channel_Interaction_Learner(self.in_channels, args.num_stages, args.node_num, self.local_image_size, self.local_image_size, args.vit_hidden_size)
        self.padding_patch = nn.Parameter(torch.zeros((1, args.vit_hidden_size)))
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(6, args.in_channels, 2, 2),
            nn.BatchNorm2d(args.in_channels),
            nn.ReLU()
        )
        self.projector = nn.Sequential(
            nn.Conv2d(self.in_channels, self.patch_embed_size, kernel_size=self.patch_size, stride=self.patch_size),
            nn.BatchNorm2d(args.patch_embed_size),
            nn.ReLU()
        )

    def image_scale(self, img_tensor):
        mean = img_tensor.mean(dim=[0,2,3]).view(1,3,1,1)
        std = img_tensor.std(dim=[0,2,3]).view(1,3,1,1)
        img_norm = (img_tensor - mean) / (std + 1e-6)
        return img_norm

    def forward(self, map_image, trajectory_images, local_patch_xids, local_patch_yids):
        '''
        Args:
            map_image (torch.Tensor): (B, H0, W0, 3)
            trajectory_images (torch.Tensor): (B, node_num, H, W, 3)
            local_patch_xids (torch.Tensor): (B, node_num, 2)
            local_patch_yids (torch.Tensor): (B, node_num, 2)
        '''
        _, H0, W0, _ = map_image.shape
        batch_size, node_num, H, W, C = trajectory_images.shape
        # static map encoder
        map_image = map_image.permute(0, 3, 1, 2).contiguous() # (B, 3, H0, W0)
        map_image = self.image_scale(map_image)
        static_map_patch_embedding = self.mask_vit(map_image)
        # static_map_patch_embedding = self.mask_vit(map_image, interpolate_pos_encoding = True).last_hidden_state
        # dynamic trajectory image encoder
        trajectory_images = trajectory_images.permute(0, 1, 4, 2, 3).contiguous() # (B, node_num, 3, H, W)
        trajectory_images = trajectory_images.view(-1, C, H, W)
        trajectory_images = self.image_scale(trajectory_images)
        trajectory_images = trajectory_images.view(batch_size, node_num, C, H, W)
        # map semantic fusion
        local_patch_xids = local_patch_xids.view(-1, 2)
        local_patch_yids = local_patch_yids.view(-1, 2)
        map_width_patch_num = W0 // self.patch_size
        edge_patch_num = self.local_image_size // self.patch_size
        local_patch_embedding_all = []
        for i in range(batch_size * node_num):
            image_patches = torch.zeros((edge_patch_num * edge_patch_num, self.patch_embed_size), dtype=torch.float32).to(map_image.device)
            if local_patch_xids[i][0] != -1:
                start_xid, end_xid = local_patch_xids[i][0], local_patch_xids[i][1]
                start_yid, end_yid = local_patch_yids[i][0], local_patch_yids[i][1]
                H_patch_num, W_patch_num = end_xid - start_xid + 1, end_yid - start_yid + 1
                local_patch_embedding = []
                for xid in range(start_xid, end_xid + 1):
                    local_patch_embed = static_map_patch_embedding[0, 1 + (xid * map_width_patch_num + start_yid) : 1 + (xid * map_width_patch_num + end_yid + 1), :]
                    if W_patch_num < edge_patch_num:
                        local_patch_embed = torch.cat([local_patch_embed, self.padding_patch.repeat(edge_patch_num - W_patch_num, 1)], dim=0)
                    local_patch_embedding.append(local_patch_embed)
                if H_patch_num < edge_patch_num:
                    local_patch_embedding.append(self.padding_patch.repeat((edge_patch_num - H_patch_num) * edge_patch_num, 1))
                image_patches = torch.cat(local_patch_embedding, dim=0)
            local_patch_embedding_all.append(image_patches)
        local_patch_embedding_all = torch.stack(local_patch_embedding_all, dim=0) # (B * node_num, patch_num, hidden_dim)
        local_patch_embedding_all = local_patch_embedding_all.permute(0, 2, 1).contiguous()
        local_patch_embedding_all = local_patch_embedding_all.view(batch_size * node_num, self.patch_embed_size, edge_patch_num, edge_patch_num)
        local_patch_embedding_all = local_patch_embedding_all.permute(0, 2, 3, 1).contiguous()
        local_patch_embedding_all = local_patch_embedding_all.view(batch_size * node_num, edge_patch_num, edge_patch_num, self.patch_size, self.patch_size, self.in_channels)
        local_patch_embedding_all = local_patch_embedding_all.permute(0, 1, 3, 2, 4, 5).contiguous()
        local_patch_embedding_all = local_patch_embedding_all.view(batch_size * node_num, edge_patch_num * self.patch_size, edge_patch_num * self.patch_size, self.in_channels)
        local_patch_embedding_all = local_patch_embedding_all.permute(0, 3, 1, 2).contiguous()
        
        # channel interaction
        inter1, inter2 = self.channel_inter_learner(trajectory_images, local_patch_embedding_all)
        decoder_image = self.image_decoder(inter2)
        vision_inter_modal = self.projector(decoder_image)
        vision_inter_modal = vision_inter_modal.view(vision_inter_modal.shape[0], vision_inter_modal.shape[1], -1).permute(0, 2, 1)
        return inter1, vision_inter_modal
