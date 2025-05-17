import torch

def _transform_x(x: torch.Tensor) -> torch.Tensor:
  """Transform the input tensor x."""
  assert len(x.shape) == 3
  bs, context_length, hidden_dim = x.shape

  x = x               # [B, N, HID_DIM]
  x = x.view(-1, 2)   # [B * N * HID_DIM/2, 2]
  x = x.t()           # [2, B * N * HID_DIM/2]
  x = x.flip(0).t()   # [B * N * HID_DIM/2, 2]
  
  y = torch.Tensor([-1., 1.])
  x = y * x           # [B * N * HID_DIM/2, 2]    
  
  x = x.contiguous()  # [B * N * HID_DIM/2, 2]
  x = x.view(bs, context_length, -1)    # [B, N, HID_DIM]

  return x

def _get_thetas(hidden_dim: int) -> torch.Tensor:
  assert hidden_dim % 2 == 0

  dinominator = torch.arange(0, hidden_dim, 2)     # [HID_DIM/2]
  dinominator = dinominator / hidden_dim        # [HID_DIM/2]
  dinominator = torch.pow(10000, dinominator)   # [HID_DIM/2]
  thetas = 1 / dinominator                      # [HID_DIM/2]
  thetas = thetas.unsqueeze(0)                  # [1, HID_DIM/2]
  thetas = thetas.repeat((2, 1))                # [2, HID_DIM/2]
  thetas = thetas.t()                           # [HID_DIM/2, 2]
  thetas = thetas.contiguous().view(-1)         # [HID_DIM]
  return thetas


def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0) -> torch.Tensor:
  """precompuate the frequency table.

  See this section for explanation:
  https://docs.google.com/document/d/1wpHwqINrmw2RT0j4WsUMnpURIF6YLqkebxVsUqCkEwQ/edit?tab=t.iy73j8yuco7x#heading=h.ok4oeb94avq
  
  Args:
    dim: the hidden dim.
    end: the max of seq length.
    theta: the base.
  """
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [HID_DIM / 2]
  t = torch.arange(end, device=freqs.device)    # [N]
  freqs = torch.outer(t, freqs)                 # [N, HID_DIM / 2]
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [N, HID_DIM / 2]
  return freqs_cis

def rope_rotate(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
  bs, n, hid_dim = x.shape              # [B, N, HID_DIM]

  x = x.view(bs, n, hid_dim//2, 2)      # [B, N, HID_DIM/2, 2]

  c = torch.complex(x[:, :, :, 0], x[:, :, :, 1]) # [B, N, HID_DIM/2]
  res = c * freqs_cis[:n]                         # [B, N, HID_DIM/2]
  real = res.real                                 # [B, N, HID_DIM/2]
  imag = res.imag                                 # [B, N, HID_DIM/2]

  res = torch.cat([real, imag], dim=-1)      # [B, N, HID_DIM]
  return res
  
class RopePosEmb(torch.nn.Module):
  def __init__(self, hidden_dim: int, context_length: int=1024):
    super().__init__()
    self.freqs_cis = precompute_freqs_cis(hidden_dim, context_length)       # [N, HID_DIM/2]
    # self.thetas = _get_thetas(hidden_dim)                               # [HID_DIM]

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return rope_rotate(x, freqs_cis=self.freqs_cis)                  # [B, N, HID_DIM]
    _, n, _ = x.shape                                            # [B, N, HID_DIM]
    
    thetas = self.thetas                                                # [HID_DIM]
    thetas = thetas.unsqueeze(0)                                        # [1, HID_DIM]
    thetas = thetas.repeat((n, 1))                                      # [N, HID_DIM]
    thetas = thetas.t()                                                 # [HID_DIM, N]
    pos = torch.arange(0, n)                                            # [N]
    thetas = pos * thetas
    thetas = thetas.t()                                                 # [N, HID_DIM]

    ret = x * torch.cos(thetas) + _transform_x(x) * torch.sin(thetas)   # [B, N, HID_DIM]
    return ret