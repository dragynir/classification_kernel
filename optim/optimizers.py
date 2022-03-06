import torch

def create_optimizer(parameters, lr):

    optimizer = torch.optim.Adam(parameters, lr=lr)

    return optimizer