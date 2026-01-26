import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class ViTSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.num_attention_heads = config.vit_num_attention_heads
        self.attention_head_size = int(config.vit_hidden_size / config.vit_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.vit_hidden_size, self.all_head_size, bias=config.vit_qkv_bias)
        self.key = nn.Linear(config.vit_hidden_size, self.all_head_size, bias=config.vit_qkv_bias)
        self.value = nn.Linear(config.vit_hidden_size, self.all_head_size, bias=config.vit_qkv_bias)

        self.dropout = nn.Dropout(config.vit_attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = context_layer

        return outputs
    
class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.vit_hidden_size, config.vit_hidden_size)
        self.dropout = nn.Dropout(config.vit_hidden_dropout_prob)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor):
        self_outputs = self.attention(hidden_states)

        attention_output = self.output(self_outputs)

        outputs = attention_output
        return outputs
    
class ViTIntermediate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.vit_hidden_size, config.vit_intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTOutput(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.vit_intermediate_size, config.vit_hidden_size)
        self.dropout = nn.Dropout(config.vit_hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states
    
class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config) -> None:
        super().__init__()
       
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.vit_hidden_size, eps=config.vit_layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.vit_hidden_size, eps=config.vit_layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        # layernorm is applied before self-attention
        self_attention_outputs = self.attention(self.layernorm_before(hidden_states))
        attention_output = self_attention_outputs

        # first residual connection
        hidden_states = attention_output + hidden_states

        # layernorm is applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = layer_output

        return outputs


class ViTEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.vit_num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor):

        for i, layer_module in enumerate(self.layer):
        
            layer_outputs = layer_module(hidden_states)

            hidden_states = layer_outputs

        return hidden_states
    
class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a Transformer.
    """

    def __init__(self, config):
        super().__init__()
        patch_size = config.vit_patch_size
        num_channels, hidden_size = config.vit_num_channels, config.vit_hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (0,0)

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor):
        batch_size, num_channels, height, width = pixel_values.shape
        self.num_patches = ((height // self.patch_size), (width // self.patch_size))
        
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2).contiguous()
        return embeddings
    
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, device, dim, edge_patch_num, x_patch_num, y_patch_num, base=10000):
      
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim = int(dim // 2)
        self.x_patch_num = x_patch_num
        self.y_patch_num = y_patch_num

        # θ = 10000^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)).to(device)
        inv_freq = inv_freq.unsqueeze(0)  # [1, dim/2]

        origin_patch_pos = torch.arange(0, edge_patch_num, dtype=torch.float32).view(1, 1, edge_patch_num).to(device)

        x_pos_ids = F.interpolate(
            origin_patch_pos,           
            scale_factor=x_patch_num / edge_patch_num,       
            mode='linear',  
            align_corners=None 
        )
        y_pos_ids = F.interpolate(
            origin_patch_pos,           
            scale_factor=y_patch_num / edge_patch_num,       
            mode='linear',  
            align_corners=None 
        )
        x_pos_ids = x_pos_ids.squeeze().unsqueeze(1)  # [num_patches, 1]
        y_pos_ids = y_pos_ids.squeeze().unsqueeze(1)  # [num_patches, 1]
        
        x_pos_embed = (x_pos_ids * inv_freq).unsqueeze(2).unsqueeze(3).repeat(1, 1, 2, 2)  # [x_patch_num, dim/2, 2, 2]
        y_pos_embed = (y_pos_ids * inv_freq).unsqueeze(2).unsqueeze(3).repeat(1, 1, 2, 2)  # [y_patch_num, dim/2, 2, 2]
        
        x_pos_embed[:,:,0,0] = torch.cos(x_pos_embed[:,:,0,0])
        x_pos_embed[:,:,1,0] = -torch.sin(x_pos_embed[:,:,1,0])
        x_pos_embed[:,:,0,1] = torch.sin(x_pos_embed[:,:,0,0])
        x_pos_embed[:,:,1,1] = torch.cos(x_pos_embed[:,:,1,0])

        y_pos_embed[:,:,0,0] = torch.cos(y_pos_embed[:,:,0,0])
        y_pos_embed[:,:,1,0] = -torch.sin(y_pos_embed[:,:,1,0])
        y_pos_embed[:,:,0,1] = torch.sin(y_pos_embed[:,:,0,0])
        y_pos_embed[:,:,1,1] = torch.cos(y_pos_embed[:,:,1,0])

        self.register_buffer("x_pos_embed", x_pos_embed)
        self.register_buffer("y_pos_embed", y_pos_embed)

    def forward(self, x):
        """
        x: [batch, patch_num, dim]
        """
        patch_num = x.size(1)
        x_rotary_embed, y_rotary_embed = x.clone(), x.clone()

        for i in range(self.x_patch_num):
            x_idx_embed = x[0, i::self.x_patch_num, :].reshape(-1, self.dim, 2).unsqueeze(2)
            x_pos = self.x_pos_embed[i:i+1, :, :, :]
            x_rotary_embed[0, i::self.x_patch_num, :] = torch.matmul(x_idx_embed, x_pos).squeeze(2).reshape(-1, self.dim * 2)

        for i in range(self.y_patch_num):
            y_idx_embed = x[0, i*self.x_patch_num:(i+1)*self.x_patch_num, :].reshape(-1, self.dim, 2).unsqueeze(2)
            y_pos = self.y_pos_embed[i:i+1, :, :, :]
            y_rotary_embed[0, i*self.x_patch_num:(i+1)*self.x_patch_num, :] = torch.matmul(y_idx_embed, y_pos).squeeze(2).reshape(-1, self.dim * 2)
 
        x_rot = x_rotary_embed + y_rotary_embed
        return x_rot
    
class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.vit_hidden_size))
        self.patch_embeddings = ViTPatchEmbeddings(config)
        self.max_num_patches = config.vit_max_num_patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.vit_hidden_size))
        self.position_embeddings = RotaryPositionEmbedding(config.device, config.vit_hidden_size, config.vit_origin_patch_num, config.map_x_patch_num, config.map_y_patch_num)
        self.dropout = nn.Dropout(config.vit_hidden_dropout_prob)
        self.config = config

    def forward(self, pixel_values: torch.Tensor):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        
        real_patch_num = self.patch_embeddings.num_patches
        embeddings = self.position_embeddings(embeddings)
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings
    
class ViTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.vit_hidden_size, eps=config.vit_layer_norm_eps)
        if config.train_model:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.vit_initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTEmbeddings):
            
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.vit_initializer_range,
            ).to(module.cls_token.dtype)
    
    def forward(self, pixel_values):
        
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values
        )
        Batch_size = embedding_output.shape[0]
        encoder_outputs = self.encoder(embedding_output)
        sequence_output = encoder_outputs
        sequence_output = self.layernorm(sequence_output)

        return sequence_output

