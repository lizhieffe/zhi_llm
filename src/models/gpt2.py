import typing

import torch
import torch.nn as nn
from src.models import gelu
from src.models import layer_norm
from src.models import rope_emb

PosEncodingType = typing.Literal["none", "abs", "sinusoidal", "rope"]


class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(config["emb_dim"], config["emb_dim"] * 4),
        gelu.GELU(),
        nn.Linear(config["emb_dim"] * 4, config["emb_dim"]),
    )

  def forward(self, x):
    return self.layers(x)


class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False, pos_emb: nn.Module = None):
    super().__init__()

    assert d_out % num_heads == 0, "d_out must be divisible by num_heads!"

    self.heads = num_heads
    self.head_dim = d_out // num_heads

    self.wk = nn.Linear(d_in, d_out, bias=qkv_bias)  # [E, H]
    self.wq = nn.Linear(d_in, d_out, bias=qkv_bias)  # [E, H]
    self.wv = nn.Linear(d_in, d_out, bias=qkv_bias)  # [E, H]
    self.droput = nn.Dropout(dropout)
    self.out_proj = nn.Linear(d_out, d_out)
    self.register_buffer('mask', torch.triu(
        torch.ones(context_length, context_length), diagonal=1))
    self.pos_emb = pos_emb

  def forward(self, x):
    """Forward.

    Args:
      x: [B, N, E]

    Returns:
      [B, N, H]
    """
    b, n, _ = x.shape

    k = self.wk(x)  # [B, N, H]
    q = self.wq(x)  # [B, N, H]
    v = self.wv(x)  # [B, N, H]

    # Assume this is the RoPE embedding.
    if self.pos_emb is not None:
      k = self.pos_emb(k)
      q = self.pos_emb(q)

    k = k.view(b, n, self.heads, self.head_dim).transpose(
        1, 2)  # [B, HEADS, N, HEAD_DIM]
    q = q.view(b, n, self.heads, self.head_dim).transpose(
        1, 2)  # [B, HEADS, N, HEAD_DIM]
    v = v.view(b, n, self.heads, self.head_dim).transpose(
        1, 2)  # [B, HEADS, N, HEAD_DIM]

    attn = q @ k.transpose(-1, -2)  # [B, HEADS, N, N]
    assert attn.shape == (b, self.heads, n, n)
    # print(f"Before causal: {attn=}")

    # [:n, :n] is to truncate to the length of input tokens.
    attn = attn.masked_fill(self.mask.bool()[:n, :n], -torch.inf)
    # print(f"After causal: {attn[0][0]=}")

    attn /= self.head_dim ** 0.5
    attn = nn.functional.softmax(attn, dim=-1)
    # print(f"After softmax: {attn[0][0]=}")
    attn = self.droput(attn)
    res = attn @ v  # [B, HEADS, N, H]
    res = res.transpose(1, 2).contiguous().view(b, n, -1)  # [B, N, H]

    res = self.out_proj(res)  # [B, N, H]

    return res


class TransformerBlock(nn.Module):

  def __init__(self, config, pos_emb: nn.Module = None):
    super().__init__()

    self.norm1 = layer_norm.LayerNorm(config["emb_dim"])
    self.norm2 = layer_norm.LayerNorm(config["emb_dim"])

    self.mha = MultiHeadAttention(
        d_in=config["emb_dim"],
        d_out=config["emb_dim"],
        context_length=config['context_length'],
        num_heads=config["n_heads"],
        dropout=config["drop_rate"],
        qkv_bias=config["qkv_bias"],
        pos_emb=pos_emb,
    )
    self.ffn = FeedForward(config)
    self.dropout = nn.Dropout(config['drop_rate'])

  def forward(self, x):
    shortcut = x
    y = self.norm1(x)
    y = self.mha(y)
    # TODO: is this needed since the MHA already has the dropout internally.
    y = self.dropout(y)
    x = y + shortcut

    shortcut = x
    y = self.norm2(x)
    y = self.ffn(y)
    y = self.dropout(y)
    y = y + shortcut

    return y


def populate_sinusoidal_pos_emb(context_length: int, hidden_dim: int) -> torch.Tensor:
  """Populate the sinusoidal positional embedding.

  Args:
    context_length: The context length.
    hidden_dim: The hidden dimension.

  Returns:
    The sinusoidal positional embedding. Shape: [context_length, hidden_dim].
  """
  pos = torch.arange(0, context_length, dtype=torch.float)  # [N]

  odd = torch.sin(pos.unsqueeze(-1) / torch.pow(10000, 2 * torch.arange(0,
                  hidden_dim, 2, dtype=torch.float) / hidden_dim))   # [N, HIDDEN/2]
  even = torch.cos(pos.unsqueeze(-1) / torch.pow(10000, 2 * torch.arange(1,
                   hidden_dim, 2, dtype=torch.float) / hidden_dim))  # [N, HIDDEN/2]

  # Interleave
  ret = torch.stack((odd, even)).view(
      2, -1).t().contiguous().view(context_length, -1)    # [N, HIDDEN]
  assert ret.shape == (context_length, hidden_dim)

  return ret


class GPTModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])

    pos_emb_type = config["pos_emb_type"]
    self.pos_emb_type = pos_emb_type
    if pos_emb_type not in typing.get_args(PosEncodingType):
      raise ValueError(f"Unknown name: {pos_emb_type}")

    self.pos_emb = None
    # self.pos_emb_val = None
    if pos_emb_type == "none":
      print("Pos embedding is disabled!")
    elif pos_emb_type == "abs":
      self.pos_emb = nn.Embedding(
          config["context_length"], config["emb_dim"])
      print("Positional encoding type: absolute")
    elif pos_emb_type == "rope":
      self.pos_emb = rope_emb.RopePosEmb(hidden_dim=config["emb_dim"])
      print("Positional encoding type: rope")
    elif pos_emb_type == "sinusoidal":
      self.register_buffer('pos_emb_val', populate_sinusoidal_pos_emb(
          config['context_length'], config["emb_dim"]))
      print("Positional encoding type: sinusoidal")
    else:
      raise ValueError(f"Unknown name: {pos_emb_type}")

    self.drop = nn.Dropout(config["drop_rate"])
    self.trf_blocks = nn.Sequential(
        *[TransformerBlock(config, pos_emb=self.pos_emb if pos_emb_type == 'rope' else None) for _ in range(config["n_layers"])])
    self.final_norm = layer_norm.LayerNorm(config["emb_dim"])
    self.out_head = nn.Linear(
        config["emb_dim"], config["vocab_size"], bias=False)

  def forward(self, x):
    _, seq_len = x.shape  # [B, N]

    if self.pos_emb and self.pos_emb_type == "abs":
      x = self.tok_emb(x) + self.pos_emb(x)
    elif self.pos_emb_val is not None:
      pos_emb_val = self.pos_emb_val[:seq_len]
      x = self.tok_emb(x) + pos_emb_val
    else:
      x = self.tok_emb(x)

    x = self.drop(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits
