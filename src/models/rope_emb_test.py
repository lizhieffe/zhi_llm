import torch
import src.models.rope_emb as rope_emb

def test_transform_x():
    x = torch.arange(0, 64, dtype=torch.float)
    x = x.view(2, 4, 8)

    print(f"input: {x.shape}")
    print(f"input: {x}")
    
    assert torch.allclose(x, torch.Tensor([[[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11., 12., 13., 14., 15.],
         [16., 17., 18., 19., 20., 21., 22., 23.],
         [24., 25., 26., 27., 28., 29., 30., 31.]],

        [[32., 33., 34., 35., 36., 37., 38., 39.],
         [40., 41., 42., 43., 44., 45., 46., 47.],
         [48., 49., 50., 51., 52., 53., 54., 55.],
         [56., 57., 58., 59., 60., 61., 62., 63.]]]))

    res = rope_emb._transform_x(x)
    assert res.shape == (2, 4, 8)
    assert torch.allclose(res, torch.Tensor([[[ -1.,   0.,  -3.,   2.,  -5.,   4.,  -7.,   6.],
         [ -9.,   8., -11.,  10., -13.,  12., -15.,  14.],
         [-17.,  16., -19.,  18., -21.,  20., -23.,  22.],
         [-25.,  24., -27.,  26., -29.,  28., -31.,  30.]],

        [[-33.,  32., -35.,  34., -37.,  36., -39.,  38.],
         [-41.,  40., -43.,  42., -45.,  44., -47.,  46.],
         [-49.,  48., -51.,  50., -53.,  52., -55.,  54.],
         [-57.,  56., -59.,  58., -61.,  60., -63.,  62.]]]))

def test_get_thetas():
    got = rope_emb._get_thetas(8)
    assert got.shape == (8,)
    assert torch.allclose(got, torch.Tensor([3.1416, 3.1416, 1.5708, 1.5708, 1.0472, 1.0472, 0.7854, 0.7854]))
    
def test_rope():
    x = torch.arange(0, 64, dtype=torch.float)
    x = x.view(2, 4, 8)     # [B, N, HID_DIM]

    rope = rope_emb.RopePosEmb(hidden_dim=8)
    res = rope(x)
    assert(res.shape, (2, 4, 8))
