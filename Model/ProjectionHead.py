import torch
import torch.nn as nn

from Model.MLP import MLP


class ProjHeadModel(nn.Module):
    def __init__(self, model, in_features, layer_name, hidden_size=4096, embedding_size=256, batch_norm_mlp=True):
        super(ProjHeadModel, self).__init__()
        self.backbone = model

        setattr(self.backbone, layer_name, nn.Identity())
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

        self.projection = MLP(in_features, embedding_size, hidden_size, batch_norm_mlp)

    def forward(self, x, return_embedding=False):
        embedding = self.backbone(x)

        if return_embedding:
            return embedding

        return self.projection(embedding)
