import torch
import torch.nn as nn
import numpy as np

from layers.patch_embed import PatchEmbed
from layers.block import TransformerBlock

class PatchNorm(nn.Module):
    """
    Patch Normalization used in DINOv2.
    Normalize each patch embedding across the embedding dimension.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x)


class VisionTransformer(nn.Module):

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        dropout_rate: float = 0.0,     # newly added
        use_patchnorm: bool = True,    # newly added
        use_layerscale: bool = True,   # newly added
    ):
        super().__init__()

        # -------- 1. Patch embedding --------
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        # Optional PatchNorm
        self.use_patchnorm = use_patchnorm
        if use_patchnorm:
            self.patchnorm = PatchNorm(embed_dim)
        else:
            self.patchnorm = nn.Identity()

        # Patch dropout
        self.patch_dropout = nn.Dropout(dropout_rate)

        # -------- 2. CLS token + positional embedding --------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # -------- 3. DropPath schedule (stochastic depth) --------
        dpr_values = np.linspace(0, drop_path_rate, depth)

        # -------- 4. Transformer blocks --------
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=float(dpr_values[i]),
                use_layerscale=use_layerscale,       # <--- I will patch this inside block
            )
            for i in range(depth)
        ])

        # -------- 5. Final LN --------
        self.norm = nn.LayerNorm(embed_dim)

        # -------- 6. Initialize Parameters --------
        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B = x.size(0)

        # 1. patchify -> [B, N, D]
        x = self.patch_embed(x)

        # 2. PatchNorm (optional)
        x = self.patchnorm(x)

        # 3. dropout
        x = self.patch_dropout(x)

        # 4. Add CLS
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)

        # 5. Add positional embed
        x = x + self.pos_embed

        # 6. Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # 7. Final norm + take CLS
        x = self.norm(x)
        return x[:, 0]
