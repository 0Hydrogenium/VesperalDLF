import torch.nn as nn
import torch


GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_size": 1024,  # Context length
    "emb_dim": 768,  # Embedding dim
    "n_heads": 12,  # Number of attn heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # QKV bias
}


"""
    GPT architecture:
        1.Layer normalization
        2.GELU activation
        3.Feed forward network
        4.Shortcut connections
"""

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg[""])

