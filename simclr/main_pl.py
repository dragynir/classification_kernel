import argparse

from pandas.core.base import NoNewAttributesMixin
import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule


# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from utils import yaml_config_hook

import os
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import shutil
import yaml
from PIL import Image

from utils import create_dataset

def create_experiment(opt):

    os.makedirs(opt.experiment_path, exist_ok=True)
    checkpoint_path = os.path.join(opt.experiment_path, 'checkpoint')
    dataset_path = os.path.join(opt.experiment_path, 'dataset')
    config_path = os.path.join(opt.experiment_path, 'config.yml')

    with open(config_path, 'w') as f:
        yaml.dump(opt, f)

    # copy dataset to experiment folder
    shutil.copytree(opt.dataset_path, dataset_path, dirs_exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    return checkpoint_path

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class ContrastiveLearning(LightningModule):
    def __init__(self, args):
        super().__init__()

        # self.hparams = args
        self.save_hyperparameters(args)

        # initialize ResNet
        self.encoder = get_resnet(self.hparams.resnet, pretrained=False)
        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.model = SimCLR(self.encoder, self.hparams.projection_dim, self.n_features)
        self.criterion = NT_Xent(
            self.hparams.batch_size, self.hparams.temperature, world_size=1
        )

    def forward(self, x_i, x_j):
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        (x_i, x_j), _ = batch
        loss = self.forward(x_i, x_j)

        self.log('loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def configure_criterion(self):
        criterion = NT_Xent(self.hparams.batch_size, self.hparams.temperature)
        return criterion

    def configure_optimizers(self):
        scheduler = None
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        elif self.hparams.optimizer == "LARS":
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
            learning_rate = 0.3 * args.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=args.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.epochs, eta_min=0, last_epoch=-1
            )
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}


if __name__ == "__main__":

    seed_everything(42)

    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "custom":
        df_folds = pd.read_csv(args.df_path, index_col=0)
        train_df = df_folds[df_folds['fold'] != 0]
        train_dataset = create_dataset(train_df, args.data_root, transforms=TransformsSimCLR(size=args.image_size))
    else:
        raise NotImplementedError

    if args.gpus == 1:
        workers = args.workers
    else:
        workers = 0


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=workers, drop_last=True)

    model_cl = ContrastiveLearning(args)

    run_name = args.project_name
    save_dir = './logdir/' + run_name
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = create_experiment(args)

    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path,
                                                    filename="{epoch:02d}_{loss:.4f}",
                                                    save_top_k=1, monitor='loss', mode='min')

    callbacks = [
                     checkpoint_callback,
                     lr_monitor_callback
                 ]

    # use wandb logger
    wandb_logger = WandbLogger(name=run_name, project=args.project_name, job_type='train',
                        save_dir=save_dir, config=args, log_model=True)
    wandb_logger.watch(model_cl)

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    trainer.sync_batchnorm=True
    trainer.fit(model_cl, train_loader)

    wandb.finish()
