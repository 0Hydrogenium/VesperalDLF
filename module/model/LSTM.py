import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lstm = nn.LSTM(cfg["model__input_dim"], cfg["model__hidden_dim"], batch_first=True)
        self.fc1 = nn.Linear(cfg["model__hidden_dim"], cfg["model__hidden_dim"])
        self.fc2 = nn.Linear(cfg["model__hidden_dim"], cfg["model__output_dim"])

    def forward(self, x):
        h, _ = self.lstm(x)
        batch_size, timestep, hidden_dim = h.shape
        h = h.reshape(-1, hidden_dim)
        h = self.fc1(h)
        h = self.fc2(h)
        h = h.reshape(timestep, batch_size, -1)
        y = h[-1]
        return y


