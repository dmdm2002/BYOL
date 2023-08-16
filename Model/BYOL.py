import torch
import torch.nn as nn

import copy

from Model.ProjectionHead import ProjHeadModel, MLP
from Utils.Losses import EMA, loss_fn


class BYOL(nn.Module):
    def __init__(self,
                 net,
                 batch_norm_mlp=True,
                 in_features=512,
                 projection_size=256,
                 projection_hidden_size=2049,
                 moving_average_decay=0.99,
                 use_momentum=True
                 ):
        super(BYOL, self).__init__()

        self.net = net
        self.online_model = ProjHeadModel(
            model=net,
            in_features=in_features,
            embedding_size=projection_size,
            hidden_size=projection_hidden_size,
            batch_norm_mlp=batch_norm_mlp
        )
        self.use_momentum = use_momentum
        self.target_model = self._get_target()
        self.target_ema_updator = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

    @torch.no_grad()
    def _get_target(self):
        return copy.deepcopy(self.online_model)

    @torch.no_grad()
    def update_moving_average(self):
        # Online network 의 weight 를 EMA 를 통해 target network 에 전해준다.
        for online_params, target_params in zip(self.online_model.parameters(), self.target_model.parameters()):
            old_weight, up_weight = target_params.data, online_params.data
            target_params.data = self.target_ema_updator.update_average(old_weight, up_weight)

    def forward(self, x1, x2=None, return_embedding=False):
        if return_embedding or (x2 is None):
            return self.online_model(x1, return_embedding=True)

        online_proj_one = self.online_model(x1)
        online_proj_two = self.online_model(x2)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_proj_one = self.target_model(x1).detach_()
            target_proj_two = self.target_model(x2).detach_()

        loss_one = loss_fn(online_pred_one, target_proj_one)
        loss_two = loss_fn(online_pred_two, target_proj_two)

        return (loss_one + loss_two).mean()