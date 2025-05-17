import torch

from src.models import gpt2
from src.models import rope_emb


def test_run():
  config = {
      "emb_dim": 768,
      "context_length": 1024,
      "num_layers": 12,
      "vocab_size": 50257,
      "qkv_bias": True,
      "drop_rate": 0.1,
      "n_layers": 12,
      "n_heads": 12,
      "pos_emb_type": "sinusoidal",
  }

  model = gpt2.GPTModel(config)

  x = torch.randint(0, config["vocab_size"], (2, config["context_length"]))
  got = model(x)
  assert got.shape == (2, config["context_length"], config["vocab_size"])


def test_pos_emb():
  config = {
      "emb_dim": 768,
      "context_length": 1024,
      "num_layers": 12,
      "vocab_size": 50257,
      "qkv_bias": True,
      "drop_rate": 0.1,
      "n_layers": 12,
      "n_heads": 12,
      "pos_emb_type": "sinusoidal",
  }

  model = gpt2.GPTModel(config)

  x = torch.randint(0, config["vocab_size"], (2, config["context_length"]))
  got = model(x)
  assert got.shape == (2, config["context_length"], config["vocab_size"])
