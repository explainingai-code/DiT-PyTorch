import torch
import torch.nn as nn
from model.patch_embed import PatchEmbedding
from model.transformer_layer import TransformerLayer
from einops import rearrange


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=temb_dim // 2,
        dtype=torch.float32,
        device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DIT(nn.Module):
    def __init__(self, im_size, im_channels, config):
        super().__init__()

        num_layers = config['num_layers']
        self.image_height = im_size
        self.image_width = im_size
        self.im_channels = im_channels
        self.hidden_size = config['hidden_size']
        self.patch_height = config['patch_size']
        self.patch_width = config['patch_size']

        self.timestep_emb_dim = config['timestep_emb_dim']

        # Number of patches along height and width
        self.nh = self.image_height // self.patch_height
        self.nw = self.image_width // self.patch_width

        # Patch Embedding Block
        self.patch_embed_layer = PatchEmbedding(image_height=self.image_height,
                                                image_width=self.image_width,
                                                im_channels=self.im_channels,
                                                patch_height=self.patch_height,
                                                patch_width=self.patch_width,
                                                hidden_size=self.hidden_size)

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.timestep_emb_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # All Transformer Layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(num_layers)
        ])

        # Final normalization for unpatchify block
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        # Scale and Shift parameters for the norm
        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)
        )

        # Final Linear Layer
        self.proj_out = nn.Linear(self.hidden_size,
                                  self.patch_height * self.patch_width * self.im_channels)

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x, t):
        # Patchify
        out = self.patch_embed_layer(x)

        # Compute Timestep representation
        # t_emb -> (Batch, timestep_emb_dim)
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.timestep_emb_dim)
        # (Batch, timestep_emb_dim) -> (Batch, hidden_size)
        t_emb = self.t_proj(t_emb)

        # Go through the transformer layers
        for layer in self.layers:
            out = layer(out, t_emb)

        # Shift and scale predictions for output normalization
        pre_mlp_shift, pre_mlp_scale = self.adaptive_norm_layer(t_emb).chunk(2, dim=1)
        out = (self.norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
               pre_mlp_shift.unsqueeze(1))

        # Unpatchify
        # (B,patches,hidden_size) -> (B,patches,channels * patch_width * patch_height)
        out = self.proj_out(out)
        out = rearrange(out, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                        ph=self.patch_height,
                        pw=self.patch_width,
                        nw=self.nw,
                        nh=self.nh)
        return out
