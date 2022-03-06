import torch

def create_sheduler(optimizer, opt):

    available_sheduler = ['reduce_lr', 'onecycle']

    if opt.sheduler not in available_sheduler:
        raise ValueError(f'Loss {opt.loss} not found, use {available_sheduler}')

    if opt.sheduler == 'onecycle':
        return 'step', torch.optim.lr_scheduler.OneCycleLR(
                max_lr=opt.lr,
                epochs=opt.max_epochs,
                optimizer=optimizer,
                steps_per_epoch=opt.steps_per_epoch,
                div_factor=opt.lr_div_factor,
                anneal_strategy=opt.anneal_strategy
            )
    elif opt.sheduler == 'reduce_lr':
        return 'epoch', torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                patience=opt.lr_patience
            )
