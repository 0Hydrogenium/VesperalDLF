import torch
import torch.nn as nn

from extra.llm.module.model.GPTModel import GPTModel


class GPT(GPTModel):
    def __init__(self, cfg):
        cfg["vocab_size"] = cfg["model__input_dim"]
        super().__init__(cfg)
        self.tok_emb = nn.Linear(cfg["vocab_size"], cfg["emb_dim"])

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))
        h = tok_embeds + pos_embeds
        h = self.drop_emb(h)
        h = self.trf_blocks(h)
        h = self.final_norm(h)
        y = self.out_head(h)
        return y

