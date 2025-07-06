import torch.nn as nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, cfg):
        super().__init__()

