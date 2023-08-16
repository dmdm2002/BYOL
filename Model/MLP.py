import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim, embedding_size=256, hidden_size=2048, batch_norm_mlp=False):
        super(MLP, self).__init__()
        norm = nn.BatchNorm1d(hidden_size) if batch_norm_mlp else nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            norm,
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, embedding_size)
        )

    def forward(self, x):
        return self.net(x)