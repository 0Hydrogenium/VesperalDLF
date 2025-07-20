import torch.nn as nn
import math
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        # 自注意力机制
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg["model__input_dim"]

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(cfg["model__negative_p"])

        # 多层Transformer编码器
        self.encoder_layers = nn.ModuleList()
        num_layers = cfg.get("model__num_layers", 3)  # 默认3层
        for _ in range(num_layers):
            self.encoder_layers.append(
                TransformerEncoderLayer(
                    d_model,
                    nhead=cfg.get("model__num_heads", 5),  # 默认5头
                    dim_feedforward=cfg.get("model__ff_dim", 512),  # FFN维度
                    dropout=cfg["model__negative_p"]
                )
            )

        # 回归输出层
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 序列池化
        self.fc = nn.Sequential(
            nn.Linear(d_model, cfg["model__hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["model__negative_p"]),
            nn.Linear(cfg["model__hidden_dim"], cfg["model__threshold_output_dim"]),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 添加位置信息
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # 通过多层编码器
        for layer in self.encoder_layers:
            x = layer(x)

        # 序列池化 (batch, seq_len, features) => (batch, features)
        x = x.permute(0, 2, 1)  # 调整为适合池化的维度
        x = self.pooling(x).squeeze(-1)

        # 输出层
        return self.fc(x)