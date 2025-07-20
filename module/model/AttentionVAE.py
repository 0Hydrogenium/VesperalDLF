import torch.nn as nn
import torch
import math


class AttentionVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg["model__input_dim"]

        # 编码器结构
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=cfg["model__negative_head"],
            batch_first=True,
            dropout=cfg["model__negative_p"] / 2
        )
        self.encoder_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=cfg["model__negative_layer_dim1"],
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # 潜在空间投影
        lstm_output_dim = 2 * cfg["model__negative_layer_dim1"]
        self.encoder_fc = nn.Sequential(
            nn.Linear(lstm_output_dim, cfg["model__negative_layer_dim2"]),
            nn.ReLU(),
            nn.LayerNorm(cfg["model__negative_layer_dim2"]),
            nn.Dropout(cfg["model__negative_p"])
        )

        # 潜在分布参数
        self.encoder_mean = nn.Linear(cfg["model__negative_layer_dim2"], cfg["model__negative_layer_dim3"])
        self.encoder_log_var = nn.Linear(cfg["model__negative_layer_dim2"], cfg["model__negative_layer_dim3"])

        # 解码器结构
        self.decoder_lstm = nn.LSTM(
            input_size=cfg["model__negative_layer_dim3"],
            hidden_size=cfg["model__negative_layer_dim2"],
            num_layers=2,
            batch_first=True
        )

        self.decoder_proj = nn.Sequential(
            nn.Linear(cfg["model__negative_layer_dim2"], cfg["model__negative_layer_dim1"]),
            nn.ReLU(),
            nn.LayerNorm(cfg["model__negative_layer_dim1"]),
            nn.Dropout(cfg["model__negative_p"]),
            nn.Linear(cfg["model__negative_layer_dim1"], embed_dim)
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()

    def reparameterize(self, mean, log_var):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def compute_kl_divergence(self, mean, log_var):
        """改进的KL散度计算 (批次平均)"""
        return -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

    def encode(self, x):
        """编码器前向传播"""
        # 自注意力机制
        attn_out, _ = self.encoder_attention(x, x, x)
        attn_out = self.norm(x + attn_out)  # 残差连接

        # 双向LSTM处理
        lstm_out, _ = self.encoder_lstm(attn_out)

        # 序列全局平均
        compressed = torch.mean(lstm_out, dim=1)

        # 特征提取
        features = self.encoder_fc(compressed)
        return self.encoder_mean(features), self.encoder_log_var(features)

    def decode(self, z, seq_length):
        """解码器前向传播"""
        # 扩展潜在向量为序列
        decoder_input = z.unsqueeze(1).repeat(1, seq_length, 1)

        # LSTM序列解码
        lstm_out, _ = self.decoder_lstm(decoder_input)

        # 逐时间步投影
        return self.decoder_proj(lstm_out)

    def forward(self, x):
        """完整前向传播"""
        batch_size, seq_len, _ = x.shape

        # 编码过程
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)

        # 解码过程
        recon = self.decode(z, seq_len)

        # 计算正则化损失
        kl_loss = self.compute_kl_divergence(mean, log_var)

        return self.norm(recon), kl_loss