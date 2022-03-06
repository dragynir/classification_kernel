import torch
from torch import nn

class MultiImageModel(nn.Module):
    def __init__(self, model):
        super(MultiImageModel, self).__init__()
        self.model = model

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.model(c_in)
        out = c_out.view(batch_size, timesteps, -1)
        return torch.mean(out, dim=1)