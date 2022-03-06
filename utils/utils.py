import os
import random
import numpy as np
import torch
import wandb
from pytorch_lightning.callbacks import Callback
import torchvision

def seed_everything(seed: int):
    '''
        Seed random for reproducibility 
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class ImagePredictionLogger(Callback):
    '''
        Predict classes and log image with label to WandB
    '''

    def __init__(self, val_samples, log_group, labels_list, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.log_group = log_group
        self.labels_list = labels_list
        self.val_imgs, self.val_labels = val_samples

    def make_grid(self, samples):
        if samples.shape > 3:
            return torchvision.utils.make_grid(samples, nrow=3)
        return samples

    def on_validation_epoch_end(self, trainer, pl_module):

        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)

        # Log the images as wandb Image
        trainer.logger.experiment.log({
            self.log_group:[wandb.Image(self.make_grid(x),
                            caption=f"Pred:{self.labels_list[pred]}, Label:{self.labels_list[pred]}")
                                for x, pred, y in zip(val_imgs[:self.num_samples],
                                                    preds[:self.num_samples],
                                                    val_labels[:self.num_samples])]
            })


def calclulate_class_weights(df):
    '''
        Calculate inverse frequency of classes 
    '''
    classes_count = df.target.value_counts().sort_index().values
    total_sum = classes_count.sum() + 1e-9
    weights = (1 - classes_count/total_sum)
    print('Calculate class weights: ', weights)
    return weights

def load_labels(labelmap_path):
    with open(labelmap_path, 'r') as f:
        labels = f.read().splitlines()
    return labels