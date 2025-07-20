import torch.nn as nn


class WeibullTwinDAELoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.gamma = cfg["model__gamma"]

    def forward(self, pred, x, mapped_y, type):
        loss = self.mse_loss(pred, x).mean(dim=2)
        if type == "negative":
            return (((1 - mapped_y) * self.gamma).reshape(-1, 1) * loss).sum(dim=0).mean()
        elif type == "positive":
            return ((mapped_y * self.gamma).reshape(-1, 1) * loss).sum(dim=0).mean()
        else:
            raise ValueError("WeibullTwinDAELoss forward type error")
