import torch

def create_optimizer(parameters, lr):
    # TODO create optimizers options

    optimizer = torch.optim.Adam(parameters, lr=lr)

    return optimizer