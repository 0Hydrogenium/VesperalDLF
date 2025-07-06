import torch.nn as nn


class MSELoss(nn.MSELoss):
    def __init__(self, cfg):
        super().__init__()

