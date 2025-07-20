import torch.nn as nn
import torch


class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder_fc1 = nn.Linear(cfg["model__input_dim"], cfg["model__layer_dim1"])
        self.encoder_fc2 = nn.Linear(cfg["model__layer_dim1"], cfg["model__layer_dim2"])
        self.encoder_mean_fc = nn.Linear(cfg["model__layer_dim2"], cfg["model__layer_dim3"])
        self.encoder_log_var_fc = nn.Linear(cfg["model__layer_dim2"], cfg["model__layer_dim3"])
        self.decoder_fc1 = nn.Linear(cfg["model__layer_dim3"], cfg["model__layer_dim2"])
        self.decoder_fc2 = nn.Linear(cfg["model__layer_dim2"], cfg["model__layer_dim1"])
        self.decoder_fc3 = nn.Linear(cfg["model__layer_dim1"], cfg["model__input_dim"])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, mean, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mean + eps * std

    def compute_kl_divergence(self, mean, log_var):
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def forward(self, x):
        h = self.encoder_fc1(x)
        h = self.relu(h)
        h = self.encoder_fc2(h)
        h = self.relu(h)
        mean = self.encoder_mean_fc(h)
        log_var = self.encoder_log_var_fc(h)

        h = self.reparameterize(mean, log_var)
        h = self.decoder_fc1(h)
        h = self.relu(h)
        h = self.decoder_fc2(h)
        h = self.relu(h)
        h = self.decoder_fc3(h)
        h = self.sigmoid(h)
        kl = self.compute_kl_divergence(mean, log_var)
        return h, kl

