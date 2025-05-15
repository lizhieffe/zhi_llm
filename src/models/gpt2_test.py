import torch

from src.models import gpt2

def test_always_passes():
    config = {
        "emb_dim": 768,
        "context_length": 1024,
        "num_layers": 12,
        "vocab_size": 50257,
        "qkv_bias": True,
        "drop_rate": 0.1,
        "n_layers": 12,
        "n_heads": 12,
    }
    
    model = gpt2.GPTModel(config)
    
    x = torch.randint(0, config["vocab_size"], (2, config["context_length"]))
    _ = model(x)