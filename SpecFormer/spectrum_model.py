import torch
import torch.nn as nn
import torch.nn.functional as F
from talking_head_attention import TalkingHeadAttention


class EncoderLayer(nn.Module):
    """Single transformer encoder layer with talking-head attention and FFN."""
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.pre_attn_layernorm = nn.LayerNorm(embed_dim)
        self.attention = TalkingHeadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.post_attn_layernorm = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Linear(embed_dim, embed_dim)
        self.ffn2 = nn.Linear(embed_dim, embed_dim)
        self.post_ffn_layernorm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """Forward pass for encoder layer.
        Args:
            x: (batch_size, seq_len, embed_dim) tensor
        Returns:
            (batch_size, seq_len, embed_dim) output tensor
        """
        # Attention block with residual connection
        x_norm = self.pre_attn_layernorm(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x_norm + self.dropout(attn_output)
        
        # FFN block with residual connection
        x_norm = self.post_attn_layernorm(x)
        ffn_output = self.ffn1(x_norm)
        ffn_output = F.relu(ffn_output)
        ffn_output = self.dropout(ffn_output)
        ffn_output = self.ffn2(ffn_output)
        x = x_norm + self.dropout(ffn_output)
        x = self.post_ffn_layernorm(x)
        
        return x


class SpectrumModel(nn.Module):
    """
    Neural network model for spectrum prediction using talking-head attention.
    Supports configurable number of encoder layers for ablation studies.
    """
    def __init__(self, token_size: int, dict_size: int, *, embed_dim: int = 256, num_heads: int = 64, seq_max_len: int = 40, penultimate_dim: int = 2048, dropout_rate: float = 0.1, num_encoder_layers: int = 1):
        super().__init__()
        self.pre_attn_proj = nn.Linear(token_size, embed_dim)

        self.global_data_projector = nn.Linear(2, embed_dim)
        self.global_data_layernorm = nn.LayerNorm(embed_dim)

        self.pos_embedding = nn.Embedding(seq_max_len, embed_dim)
        
        self.dropout = nn.Dropout(dropout_rate)

        # Stack of encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, dropout_rate)
            for _ in range(num_encoder_layers)
        ])

        self.penultimate_proj = nn.Linear(embed_dim, penultimate_dim)
        self.penultimate_norm = nn.LayerNorm(penultimate_dim)

        self.output_proj = nn.Linear(penultimate_dim, dict_size)

    def forward(self, x, charges, NCEs):
        """
        Forward pass for spectrum prediction.
        Args:
            x: (batch_size, src_len, token_size) input sequence tensor
            charges: (batch_size,) tensor
            NCEs: (batch_size,) tensor
        Returns:
            (batch_size, dict_size) output tensor
        """
        # x shape: (batch_size, src_len, token_size)
        batch_size, src_len, token_size = x.shape
        device = x.device  # Ensure all new tensors are on the same device as input

        x = self.pre_attn_proj(x)

        global_data = torch.stack([charges, NCEs], dim=1).float().to(device)
        global_embed = self.global_data_projector(global_data)
        global_embed = self.global_data_layernorm(global_embed)
        x = x + global_embed.unsqueeze(1) # global_embed is broadcasted to (batch_size, src_len, embed_dim)

        pos_src = torch.arange(0, src_len, device=device).unsqueeze(1).expand(src_len, batch_size)
        x = x + self.pos_embedding(pos_src).transpose(0, 1)

        # Apply dropout to the combined embeddings
        x = self.dropout(x)

        # Pass through all encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # Penultimate block
        x = self.penultimate_proj(x)
        x = self.penultimate_norm(x)
        x = F.relu(x)
        x = self.dropout(x) # Dropout after final activation before output layer

        # Output block
        x = self.output_proj(x)
        x = torch.sigmoid(x)
        x = x.mean(dim=1)
        
        return x