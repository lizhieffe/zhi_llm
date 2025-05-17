import torch


def _precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
  """precompuate the frequency table.

  See this section for explanation:
  https://docs.google.com/document/d/1wpHwqINrmw2RT0j4WsUMnpURIF6YLqkebxVsUqCkEwQ/edit?tab=t.iy73j8yuco7x#heading=h.ok4oeb94avq

  Args:
    dim: the hidden dim.
    end: the max of seq length.
    theta: the base.
  """
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                 [: (dim // 2)].float() / dim))  # [HID_DIM / 2]
  t = torch.arange(end, device=freqs.device)    # [N]
  freqs = torch.outer(t, freqs)                 # [N, HID_DIM / 2]
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [N, HID_DIM / 2]
  return freqs_cis


def _rotate(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
  bs, n, hid_dim = x.shape              # [B, N, HID_DIM]

  x = x.view(bs, n, hid_dim // 2, 2)      # [B, N, HID_DIM/2, 2]

  c = torch.complex(x[:, :, :, 0], x[:, :, :, 1])  # [B, N, HID_DIM/2]
  res = c * freqs_cis[:n]                         # [B, N, HID_DIM/2]
  real = res.real                                 # [B, N, HID_DIM/2]
  imag = res.imag                                 # [B, N, HID_DIM/2]

  res = torch.cat([real, imag], dim=-1)           # [B, N, HID_DIM]
  return res


class RopePosEmb(torch.nn.Module):
  def __init__(self, hidden_dim: int, context_length: int = 1024):
    super().__init__()
    self.register_buffer('freqs_cis', _precompute_freqs_cis(
        hidden_dim, context_length))       # [N, HID_DIM/2]                       # [HID_DIM]

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # [B, N, HID_DIM]
    return _rotate(x, freqs_cis=self.freqs_cis)
