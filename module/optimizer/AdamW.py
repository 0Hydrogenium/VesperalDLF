import torch


class AdamW(torch.optim.AdamW):
    def __init__(self, cfg, model):
        super().__init__(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"]
        )
