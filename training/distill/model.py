"""Student encoder — the model trained to mimic the teacher.

Architecture (micro tier, locked from POC scaled run):

    input_ids  (batch, seq_len=128)
        ↓ Embedding (vocab_size × d_model=256, padding_idx=0)
        ↓ TransformerLayer × n_layers=2
        ↓ LayerNorm (over d_model)
        ↓ Mean pool over non-padding positions
        ↓ Linear projection (d_model=256 → output_dim=384)
        ↓ L2 normalize
    embedding  (batch, 384)   ← compared against teacher via cosine sim

Phase 2 runs this in pure float32. Phase 3 (QAT) wraps the Q/K/V/W_out and
fc1/fc2 Linear layers with BitLinear via `replace_modules`; the projection
layer is excluded from the swap and stays fp32 — it's the bridge between
two dimensional spaces and is quantization-sensitive (see
docs/training/postmortem-bitlinear-asymmetry.md).

Why explicit Q/K/V (not nn.MultiheadAttention): BitLinear's `replace_modules`
needs to find individual nn.Linear instances. nn.MultiheadAttention bundles
Q/K/V into a single Parameter that the swap can't reach.

Legacy checkpoints omit positions. The second bilingual pilot enables learned
positions to distinguish role reversals that a permutation-invariant model cannot.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with explicit Q/K/V projections."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.W_q   = nn.Linear(d_model, d_model, bias=False)
        self.W_k   = nn.Linear(d_model, d_model, bias=False)
        self.W_v   = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                    # (B, T, d_model)
        mask: torch.Tensor | None = None,   # (B, T) — 1 real, 0 pad
    ) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale  = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if mask is not None:
            # Mask padding KEYS (column j) so softmax assigns them zero weight.
            pad_mask = (mask == 0).unsqueeze(1).unsqueeze(2)   # (B, 1, 1, T)
            scores   = scores.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                              # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_out(out)


class FeedForward(nn.Module):
    """d_model → ffn_dim → d_model, with GELU. Standard 4× expansion."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(d_model, ffn_dim)
        self.fc2     = nn.Linear(ffn_dim, d_model)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerLayer(nn.Module):
    """Pre-LN encoder block: norm → sublayer → residual."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn    = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff      = FeedForward(d_model, ffn_dim, dropout)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class StudentEncoder(nn.Module):
    """Tokens → 384-dim normalized sentence embedding."""

    def __init__(
        self,
        vocab_size:  int   = 30_522,
        d_model:     int   = 256,
        n_layers:    int   = 2,
        n_heads:     int   = 4,
        ffn_dim:     int   = 1_024,
        output_dim:  int   = 384,
        dropout:     float = 0.1,
        padding_idx: int   = 0,
        position_embeddings: bool = False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(128, d_model) if position_embeddings else None
        self.layers    = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm       = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, output_dim)   # excluded from BitLinear swap

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.002)
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].fill_(0.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids:      torch.Tensor,               # (B, T)
        attention_mask: torch.Tensor | None = None, # (B, T)
    ) -> torch.Tensor:
        """Returns (B, output_dim) L2-normalized embedding."""
        x = self.embedding(input_ids)
        if self.position_embedding is not None:
            x = x + self.position_embedding.weight[:input_ids.shape[1]]

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)

        if attention_mask is not None:
            mask   = attention_mask.unsqueeze(-1).float()       # (B, T, 1)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        projected = self.projection(pooled)
        return F.normalize(projected, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
