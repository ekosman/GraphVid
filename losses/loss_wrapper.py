from torch import nn


class LossWrapper(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, x, y):
        return self.loss(x, y),
