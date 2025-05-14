import torch
import torch.nn as nn

class LayerNorm(nn.Module):
  def __init__(self, emb_dim: int):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self, x):
    # x [B, N, H_DIM]
    mean = x.mean(dim=-1, keepdim=True) # [B, N, 1]
    #  in the variance calculation, we divide by the number of inputs n in the variance formula. This approach does
    # not apply Bessel’s correction, which typically uses n – 1 instead of n in the denomi-
    # nator to adjust for bias in sample variance estimation. This decision results in a so-
    # called biased estimate of the variance. For LLMs, where the embedding dimension n
    # is significantly large, the difference between using n and n – 1 is practically negligible.
    # I chose this approach to ensure compatibility with the GPT-2 model’s normalization
    # layers and because it reflects TensorFlow’s default behavior, which was used to
    # implement the original GPT-2 model. Using a similar setting ensures our method is
    # compatible with the pretrained weights we will load in chapter 6.
    var = x.var(dim=-1, keepdim=True, unbiased=False)   # [B, N, 1]
    normalized = (x - mean) / torch.sqrt(var + self.eps) # [B, N, H_DIM]
    return self.scale * normalized + self.shift