import torch.nn as nn
import torch


class AttentionVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder_attention = nn.MultiheadAttention(embed_dim=cfg["model__input_dim"], num_heads=cfg["model__negative_head"], batch_first=True)
        self.encoder_fc1 = nn.Linear(cfg["model__input_dim"], cfg["model__negative_layer_dim1"])
        self.encoder_fc2 = nn.Linear(cfg["model__negative_layer_dim1"], cfg["model__negative_layer_dim2"])
        self.encoder_mean_fc = nn.Linear(cfg["model__negative_layer_dim2"], cfg["model__negative_layer_dim3"])
        self.encoder_log_var_fc = nn.Linear(cfg["model__negative_layer_dim2"], cfg["model__negative_layer_dim3"])
        self.decoder_fc1 = nn.Linear(cfg["model__negative_layer_dim3"], cfg["model__negative_layer_dim2"])
        self.decoder_fc2 = nn.Linear(cfg["model__negative_layer_dim2"], cfg["model__negative_layer_dim1"])
        self.decoder_fc3 = nn.Linear(cfg["model__negative_layer_dim1"], cfg["model__input_dim"])
        self.dropout = nn.Dropout(p=cfg["model__negative_p"])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pass
