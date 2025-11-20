import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer with QK-Norm (DINOv2-style).
    This keeps your original structure but adds:
        - q_norm per head
        - k_norm per head
    which greatly stabilizes SSL training and improves performance.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_qk_norm = use_qk_norm

        # ======= combined qkv projection =======
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)

        # ======= per-head LayerNorm (QK-Norm) =======
        if use_qk_norm:
            # normalize along the head_dim
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        # ======= output projection =======
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        return: [B, N, D]
        """
        B, N, D = x.shape

        # ========== Step 1: project to qkv ==========
        qkv = self.qkv(x)                                   # [B, N, 3D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                         # [B, N, H, Hd]

        # ========== Step 2: move heads ==========
        # -> [B, H, N, Hd]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ========== Step 3: QK-Norm ==========
        # LayerNorm on Hd dim for each head separately
        if self.use_qk_norm:
            # q: [B, H, N, Hd] → apply LN over last dim
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ========== Step 4: scaled dot-product attention ==========
        attn = (q @ k.transpose(-2, -1)) * self.scale       # [B, H, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # ========== Step 5: attention output ==========
        out = attn @ v                                      # [B, H, N, Hd]

        # merge heads: → [B, N, D]
        out = out.transpose(1, 2).reshape(B, N, D)

        # ========== Step 6: final projection ==========
        out = self.proj(out)
        out = self.proj_drop(out)

        return out
