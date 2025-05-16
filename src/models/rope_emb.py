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

  dinominator = torch.arange(0, hidden_dim / 2) # [HID_DIM/2]
  thetas = torch.pi / (dinominator + 1)         # [HID_DIM/2]
  thetas = thetas.unsqueeze(0)                  # [1, HID_DIM/2]
  thetas = thetas.repeat((2, 1))                # [2, HID_DIM/2]
  thetas = thetas.t()                           # [HID_DIM/2, 2]
  thetas = thetas.contiguous().view(-1)         # [HID_DIM]
  return thetas

class RopePosEmb(torch.nn.Module):
  def __init__(self, hidden_dim: int):
    super().__init__()
    self.thetas = _get_thetas(hidden_dim)                               # [HID_DIM]

  def forward(self, x: torch.Tensor) -> torch.Tensor:
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