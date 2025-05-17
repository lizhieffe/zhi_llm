import torch
import src.models.rope_emb as rope_emb


def test_rope():
    x = torch.arange(0, 64, dtype=torch.float)
    x = x.view(2, 4, 8)     # [B, N, HID_DIM]

    rope = rope_emb.RopePosEmb(hidden_dim=8)
    res = rope(x)
    assert (res.shape, (2, 4, 8))


def test_precompute_freqs_cis():
    got = rope_emb._precompute_freqs_cis(dim=4, end=4)
    assert got.shape == (4, 2)


def test_rope_rotate():
    x = torch.arange(0, 64, dtype=torch.float)
    x = x.view(2, 4, 8)     # [B, N, HID_DIM]

    freqs_cis = rope_emb._precompute_freqs_cis(dim=8, end=4)
    res = rope_emb._rotate(x, freqs_cis)
    assert (res.shape == (2, 4, 8))
