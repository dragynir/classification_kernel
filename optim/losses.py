import torch


def create_loss(opt, weights=None):

    available_losses = ['cross_entropy']

    if opt.loss not in available_losses:
        raise ValueError(f'Loss {opt.loss} not found, use {available_losses}')

    if opt.loss == 'cross_entropy':
        return torch.nn.CrossEntropyLoss(weight=weights)
