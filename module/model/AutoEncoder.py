import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder_fc1 = nn.Linear(cfg["model__input_dim"], cfg["model__layer_dim1"])
        self.encoder_fc2 = nn.Linear(cfg["model__layer_dim1"], cfg["model__layer_dim2"])
        self.encoder_fc3 = nn.Linear(cfg["model__layer_dim2"], cfg["model__layer_dim3"])
        self.decoder_fc1 = nn.Linear(cfg["model__layer_dim3"], cfg["model__layer_dim2"])
        self.decoder_fc2 = nn.Linear(cfg["model__layer_dim2"], cfg["model__layer_dim1"])
        self.decoder_fc3 = nn.Linear(cfg["model__layer_dim1"], cfg["model__input_dim"])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.encoder_fc1(x)
        h = self.relu(h)
        h = self.encoder_fc2(h)
        h = self.relu(h)
        h = self.encoder_fc3(h)
        h = self.relu(h)

        h = self.decoder_fc1(h)
        h = self.relu(h)
        h = self.decoder_fc2(h)
        h = self.relu(h)
        h = self.decoder_fc3(h)
        y = self.sigmoid(h)

        return y
