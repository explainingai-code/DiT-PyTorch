import torch.nn as nn
from model.attention import Attention


class TransformerLayer(nn.Module):
    r"""
    Transformer block which is just doing the following based on VIT
        1. LayerNorm followed by Attention
        2. LayerNorm followed by Feed forward Block
        Both these also have residuals added to them

        For DiT we additionally have
        1. Layernorm mlp to predict layernorm affine parameters from
        2. Same Layernorm mlp to also predict scale parameters for outputs
            of both mlp/attention prior to residual connection.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']

        ff_hidden_dim = 4 * self.hidden_size

        # Layer norm for attention block
        self.att_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        self.attn_block = Attention(config)

        # Layer norm for mlp block
        self.ff_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        self.mlp_block = nn.Sequential(
            nn.Linear(self.hidden_size, ff_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ff_hidden_dim, self.hidden_size),
        )

        # Scale Shift Parameter predictions for this layer
        # 1. Scale and shift parameters for layernorm of attention (2 * hidden_size)
        # 2. Scale and shift parameters for layernorm of mlp (2 * hidden_size)
        # 3. Scale for output of attention prior to residual connection (hidden_size)
        # 4. Scale for output of mlp prior to residual connection (hidden_size)
        # Total 6 * hidden_size
        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.mlp_block[0].weight)
        nn.init.constant_(self.mlp_block[0].bias, 0)
        nn.init.xavier_uniform_(self.mlp_block[-1].weight)
        nn.init.constant_(self.mlp_block[-1].bias, 0)

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

    def forward(self, x, condition):
        scale_shift_params = self.adaptive_norm_layer(condition).chunk(6, dim=1)
        (pre_attn_shift, pre_attn_scale, post_attn_scale,
         pre_mlp_shift, pre_mlp_scale, post_mlp_scale) = scale_shift_params
        out = x
        attn_norm_output = (self.att_norm(out) * (1 + pre_attn_scale.unsqueeze(1))
                            + pre_attn_shift.unsqueeze(1))
        out = out + post_attn_scale.unsqueeze(1) * self.attn_block(attn_norm_output)
        mlp_norm_output = (self.ff_norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
                           pre_mlp_shift.unsqueeze(1))
        out = out + post_mlp_scale.unsqueeze(1) * self.mlp_block(mlp_norm_output)
        return out
