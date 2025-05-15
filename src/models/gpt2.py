import torch
import torch.nn as nn
from src.models import gelu
from src.models import layer_norm

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
    b, n, _ = x.shape

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

class TransformerBlock(nn.Module):

  def __init__(self, config):
    super().__init__()

    self.norm1 = layer_norm.LayerNorm(config["emb_dim"])
    self.norm2 = layer_norm.LayerNorm(config["emb_dim"])
    self.mha = MultiHeadAttention(
        d_in=config["emb_dim"],
        d_out=config["emb_dim"],
        context_length=config['context_length'],
        num_heads=config["n_heads"],
        dropout=config["drop_rate"],
        qkv_bias=config["qkv_bias"]
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
  
class GPTModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
    self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
    self.drop = nn.Dropout(config["drop_rate"])
    self.trf_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["n_layers"])])
    self.final_norm = layer_norm.LayerNorm(config["emb_dim"])
    self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

  def forward(self, x):
    bs, seq_len = x.shape
    tok_emb = self.tok_emb(x)
    pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device))
    x = tok_emb + pos_emb
    x = self.drop(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits