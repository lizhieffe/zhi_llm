import torch
import torch.nn as nn
# from models.layer_norm import LayerNorm
# from src.models.layer_norm import LayerNorm
# import gelu
from . import gelu

from src.models.gelu import GELU

# from . import gelu
# from gelu import GELU

# from models import gelu

class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(config["emb_dim"], config["emb_dim"] * 4),
        GELU(),
        nn.Linear(config["emb_dim"] * 4, config["emb_dim"]),
    )

  def forward(self, x):
    return self.layers(x)

class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
    super().__init__()

    assert d_out % num_heads == 0, "d_out must be divisible by num_heads!"

    self.heads = num_heads
    self.head_dim = d_out // num_heads

    self.wk = nn.Linear(d_in, d_out, bias=qkv_bias) # [E, H]
    self.wq = nn.Linear(d_in, d_out, bias=qkv_bias) # [E, H]
    self.wv = nn.Linear(d_in, d_out, bias=qkv_bias) # [E, H]
    self.droput = nn.Dropout(dropout)
    self.out_proj = nn.Linear(d_out, d_out)
    self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

  def forward(self, x):
    """Forward.

    Args:
      x: [B, N, E]

    Returns:
      [B, N, H]
    """
    b, n, d_in = x.shape

    k = self.wk(x) # [B, N, H]
    q = self.wq(x) # [B, N, H]
    v = self.wv(x) # [B, N, H]

    k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2) # [B, HEADS, N, HEAD_DIM]
    q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2) # [B, HEADS, N, HEAD_DIM]
    v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2) # [B, HEADS, N, HEAD_DIM]

    attn = q @ k.transpose(-1, -2) # [B, HEADS, N, N]
    assert attn.shape == (b, self.heads, n, n)
    # print(f"Before causal: {attn=}")

    # [:n, :n] is to truncate to the length of input tokens.
    attn = attn.masked_fill(self.mask.bool()[:n, :n], -torch.inf)
    # print(f"After causal: {attn[0][0]=}")

    attn /= self.head_dim ** 0.5
    attn = nn.functional.softmax(attn, dim=-1)
    # print(f"After softmax: {attn[0][0]=}")
    attn = self.droput(attn)
    res = attn @ v # [B, HEADS, N, H]
    res = res.transpose(1, 2).contiguous().view(b, n, -1) # [B, N, H]

    res = self.out_proj(res)  # [B, N, H]

    return res