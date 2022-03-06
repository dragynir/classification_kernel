import os
import random
import numpy as np
import pandas as pd
import torch
import shutil
import wandb
import yaml

from pytorch_lightning.loggers import WandbLogger
from torchmetrics import F1, AveragePrecision
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor


from data import create_dataloader, create_dataset, create_transforms
from data import NightTransforms
from models import create_model
from optim import create_optimizer, create_loss
from sheduler import create_sheduler
from utils import seed_everything
from utils import ImagePredictionLogger
from utils import calclulate_class_weights


# fix random for experiments reproducibility
seed = 42
seed_everything(seed) 


class Model(pl.LightningModule):
    # TODO add label smoothing to loss
    def __init__(self, train_dataset, val_dataset, opt, *args, **kwargs):
        super().__init__()
        self.opt = opt
        self.kwargs = kwargs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.net = create_model(opt.architecture, opt.pretrained, opt.freeze_encoder, opt.num_classes)
        self.f1_metric = F1(num_classes=opt.num_classes, average='macro')

        self.ap_metric = AveragePrecision(num_classes=opt.num_classes, average='macro')

        # create dataloaders to log image in WandB
        self.log_val_dataloader = create_dataloader(self.val_dataset, self.opt.log_batch_size,
                                 self.opt.num_workers, shuffle=True)
        self.log_train_dataloader = create_dataloader(self.train_dataset, self.opt.log_batch_size,
                                 self.opt.num_workers, shuffle=True)

        self.loss_function = self.configure_loss()
        self.lr = opt.lr


    def forward(self, x):
        return self.net(x)

    def configure_loss(self):
        self.weights = self.kwargs.get('class_weights') if 'class_weights' in self.kwargs else None
        if self.weights is not None:
            self.weights = torch.FloatTensor(self.weights)
        return create_loss(self.opt, self.weights)

    def configure_optimizers(self):

        # set lr from lr_finder
        self.opt.lr = (self.lr or self.learning_rate)

        optimizer = create_optimizer(self.parameters(), self.opt.lr)

        interval, scheduler = create_sheduler(optimizer, self.opt)

        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'val_loss',
                        'name': 'learning_rate',
                        'interval': interval,
                        'frequency': 1}

        return [optimizer], [lr_scheduler]

    def step(self, batch):

        x, y  = batch

        y_hat = self(x)

        loss  = self.loss_function(y_hat, y)

        return loss, y, F.softmax(y_hat, dim=1)

    def training_step(self, batch, batch_nb):

        loss, y, y_hat = self.step(batch)

        f1_score = self.f1_metric(y_hat, y)
        ap_score = self.ap_metric(y_hat, y)

        # log train loss and metrics to WandB
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_f1_score', f1_score, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_ap_score', ap_score, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        loss, y, y_hat = self.step(batch)
        return {'val_loss': loss,
                'y': y.detach(), 'y_hat': y_hat.detach()}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])

        f1_score = self.f1_metric(y_hat, y)
        ap_score = self.ap_metric(y_hat, y)

        # log val loss and metrics to WandB
        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.log('val_f1_score', f1_score, prog_bar=True, sync_dist=True)
        self.log('val_ap_score', ap_score, prog_bar=True, sync_dist=True)

        return {'val_loss': avg_loss}

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.opt.batch_size,
                                 self.opt.num_workers, shuffle=True)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.opt.batch_size,
                                 self.opt.num_workers, shuffle=False)


def create_experiment(opt):
    '''
        Create experiment folder, copy dataset files and config file
    '''
    os.makedirs(opt.experiment_path, exist_ok=True)
    checkpoint_path = os.path.join(opt.experiment_path, 'checkpoint')
    dataset_path = os.path.join(opt.experiment_path, 'dataset')
    config_path = os.path.join(opt.experiment_path, 'config.yml')

    with open(config_path, 'w') as f:
        yaml.dump(opt, f)

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    # copy dataset to experiment folder
    shutil.copytree(opt.dataset_path, dataset_path)
    os.makedirs(checkpoint_path, exist_ok=True)

    return checkpoint_path


def get_domain_transforms(labelmap_path, night_label_path, domain):
    
    if not night_label_path:
        return None

    with open(labelmap_path, 'r') as f:
        labels = f.read().splitlines()

    with open(night_label_path, 'r') as f:
        n_labels = f.read().splitlines()

    ind = [labels.index(nl) for nl in n_labels]

    return domain(ind)


def train(df_folds: pd.DataFrame, fold_number, opt):
    '''
        df: k-fold dataframe
    '''

    run_name = opt.project_name + '-{fold_number}' if opt.kfold else opt.project_name
    save_dir = './logdir/' + run_name
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = create_experiment(opt)

    val_transforms = create_transforms(opt, 'val')

    train_transforms = create_transforms(opt, 'train', post_transforms=None)

    train_df = df_folds[df_folds['fold'] != fold_number]

    train_dataset = create_dataset(train_df, opt.data_root,transforms=train_transforms)
    opt.steps_per_epoch = int(len(train_df)/opt.batch_size)


    val_df = df_folds[df_folds['fold'] == fold_number]
    val_dataset = create_dataset(val_df, opt.data_root, transforms=val_transforms)

    class_weights = None
    if opt.class_weights:
        class_weights = calclulate_class_weights(train_df)

    model = Model(train_dataset, val_dataset, opt, class_weights=class_weights)

    # use wandb logger
    wandb_logger = WandbLogger(name=run_name, project=opt.project_name, job_type='train',
                        save_dir=save_dir, config=opt, log_model=True)

    wandb_logger.watch(model)

    # create callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=opt.early_stop_patience,
        verbose=False,
        mode='min'
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path,
                                                    filename="{epoch:02d}_{val_loss:.4f}",
                                                    save_top_k=1, monitor='val_loss', mode='min')

    val_samples = next(iter(model.log_val_dataloader))
    train_samples = next(iter(model.log_train_dataloader))

    callbacks = [
                     # log images to WandB
                     ImagePredictionLogger(val_samples, 'val_examples'),
                     ImagePredictionLogger(train_samples, 'train_examples'),

                     early_stop_callback,
                     checkpoint_callback,
                     lr_monitor_callback
                 ]

    trainer = pl.Trainer(
        gpus=opt.gpus, # use first and second gpus (check out Pytorch Lightning docs)
        precision=opt.precision,
        max_epochs=opt.max_epochs,
        logger=wandb_logger, # set logger to allow logging to WandB
        callbacks=callbacks,
        auto_lr_find=opt.auto_lr_find
    )

    if opt.auto_lr_find:
        trainer.tune(model)

    trainer.fit(model)

    print('Finish traning...')

    trainer.validate(model)

    print('Finish validation...')

    wandb.finish()
