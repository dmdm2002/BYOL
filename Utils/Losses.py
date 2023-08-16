import torch
import torch.nn.functional as F


class EMA:
    def __init__(self, alpha):
        self.alpha = alpha

    def update_average(self, old, new):
        if old is None:
            return new

        return old * self.alpha + (1 - self.alpha) * new


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    return 2-2 * (x + y).sum(dim=-1)