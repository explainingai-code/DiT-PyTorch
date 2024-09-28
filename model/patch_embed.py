import torch
import torch.nn as nn
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_patch_position_embedding(pos_emb_dim, grid_size, device):
    assert pos_emb_dim % 4 == 0, 'Position embedding dimension must be divisible by 4'
    grid_size_h, grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)

    # grid_h_positions -> (Number of patch tokens,)
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=pos_emb_dim // 4,
        dtype=torch.float32,
        device=device) / (pos_emb_dim // 4))
    )

    grid_h_emb = grid_h_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)
    # grid_h_emb -> (Number of patch tokens, pos_emb_dim // 2)

    grid_w_emb = grid_w_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)
    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    # pos_emb -> (Number of patch tokens, pos_emb_dim)
    return pos_emb


class PatchEmbedding(nn.Module):
    r"""
    Layer to take in the input image and do the following:
        1.  Transform grid of image patches into a sequence of patches.
            Number of patches are decided based on image height,width and
            patch height, width.
        2. Add positional embedding to the above sequence
    """

    def __init__(self,
                 image_height,
                 image_width,
                 im_channels,
                 patch_height,
                 patch_width,
                 hidden_size):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.im_channels = im_channels

        self.hidden_size = hidden_size

        self.patch_height = patch_height
        self.patch_width = patch_width

        # Input dimension for Patch Embedding FC Layer
        patch_dim = self.im_channels * self.patch_height * self.patch_width
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, self.hidden_size)
        )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.patch_embed[0].weight)
        nn.init.constant_(self.patch_embed[0].bias, 0)

    def forward(self, x):
        grid_size_h = self.image_height // self.patch_height
        grid_size_w = self.image_width // self.patch_width

        # B, C, H, W -> B, (Patches along height * Patches along width), Patch Dimension
        # Number of tokens = Patches along height * Patches along width
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                        ph=self.patch_height,
                        pw=self.patch_width)

        # BxNumber of tokens x Patch Dimension -> B x Number of tokens x Transformer Dimension
        out = self.patch_embed(out)

        # Add 2d sinusoidal position embeddings
        pos_embed = get_patch_position_embedding(pos_emb_dim=self.hidden_size,
                                                 grid_size=(grid_size_h, grid_size_w),
                                                 device=x.device)
        out += pos_embed
        return out

