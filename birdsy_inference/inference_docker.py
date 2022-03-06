from efficientnet_pytorch import EfficientNet
from torch import nn
import pytorch_lightning as pl
import torch
import yaml
from addict import Dict
import cv2
from torch.nn import functional as F
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import numpy as np


class Model(pl.LightningModule):

    def __init__(self, num_classes, architecture, *args, **kwargs):
        super().__init__()
        self.net = EfficientNet.from_name(architecture)
        self.net._fc = nn.Linear(in_features=self.net._fc.in_features, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.net(x)


class ClassificationPredictor():

    def __init__(self, cfg_path, labels_path, checkpoint_path):
        self.opt = ClassificationPredictor.__load_cfg(cfg_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ClassificationPredictor.__load_model(self.opt, checkpoint_path, self.device)
        self.labels = ClassificationPredictor.__load_labels(labels_path)
        self.transforms = ClassificationPredictor.__create_transforms(self.opt)

    @staticmethod
    def __load_cfg(cfg_path):
        with open(cfg_path, 'r') as f:
            opt = Dict(yaml.safe_load(f))
        return opt

    @staticmethod
    def __load_labels(labels_path):
        with open(labels_path, 'r') as f:
            labels = f.read().splitlines()
        print(f'Labels {labels_path} loaded')
        return labels

    @staticmethod
    def __load_model(model_cfg, checkpoint_path, device):
        model = Model(model_cfg.num_classes, model_cfg.architecture)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])
        model.eval()
        model.to(device)
        print(f'Model {checkpoint_path} loaded')
        return model

    @staticmethod
    def __create_transforms(opt):
        transforms = A.Compose([
            A.Resize(height=opt.resolution, width=opt.resolution, p=1.0),
            A.Normalize(),
            ToTensorV2(),
        ], p=1.0)
        print('Transforms completed')
        return transforms

    @staticmethod
    def __indexes2labels(labels, indexes):
        return list(map(lambda x: labels[x], indexes))

    def predict_image(self, image, top_k=4):
        
        sample = self.transforms(image=image)
        image_tr = sample['image']
        image_tr = torch.unsqueeze(image_tr, dim=0)

        with torch.no_grad():
            logits = self.model(image_tr.to(self.device))
            probs = F.softmax(logits, dim=1)
            top_probs, top_labels = torch.topk(probs, top_k, dim=1)
            y_pred = top_labels.squeeze().cpu().numpy()
            probs = top_probs.squeeze().cpu().numpy()

        pred_labels = ClassificationPredictor.__indexes2labels(self.labels, y_pred)
        results = {lb: p for lb, p in zip(pred_labels, probs)}

        return results


def load_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        opt = Dict(yaml.safe_load(f))
    return opt


def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels = f.read().splitlines()
    return labels


def load_model(model_cfg, checkpoint_path):
    model = Model(model_cfg.num_classes, model_cfg.architecture)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()
    return model


def create_transforms(opt):
    transforms = A.Compose([
        A.Resize(height=opt.resolution, width=opt.resolution, p=1.0),
        A.Normalize(),
        ToTensorV2(),
    ], p=1.0)
    return transforms


def indexes2labels(labels, indexes):
    return list(map(lambda x: labels[x], indexes))


def process_image(image: Image, labels, model, transforms, device, top_k=4):
    np_image = np.asarray(image)
    bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    sample = transforms(image=bgr_image)
    image_tr = sample['image']
    image_tr = torch.unsqueeze(image_tr, dim=0)

    with torch.no_grad():
        logits = model(image_tr.to(device))
        probs = F.softmax(logits, dim=1)
        top_probs, top_labels = torch.topk(probs, top_k, dim=1)
        y_pred = top_labels.squeeze().cpu().numpy()
        probs = top_probs.squeeze().cpu().numpy()

    pred_labels = indexes2labels(labels, y_pred)
    results = {lb: p for lb, p in zip(pred_labels, probs)}

    return results


def predict_image(image_path, cfg_path, labels_path, checkpoint_path):
    opt = load_cfg(cfg_path)
    model = load_model(opt, checkpoint_path)
    transforms = create_transforms(opt)
    labels = load_labels(labels_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    pil_image = Image.open(image_path)

    results = process_image(pil_image, labels, model, transforms, device, top_k=4)

    return results


if __name__ == '__main__':
    image_path = '/home/mborisov/CLM/test_random_images/EPY-dHgXkAA7z17.jpg'
    cfg_path = '../config/base.yml'
    checkpoint_path = '/home/mborisov/CLM/checkpoints/merge_groups_20_10_2021/epoch=21_val_loss=0.0823.ckpt'
    labels_path = '../datasets/birdsy_labels.txt'

    # use preloaded model
    predictor = ClassificationPredictor(cfg_path, labels_path, checkpoint_path)
    pil_image = Image.open(image_path)
    print(predictor.predict_image(pil_image))

    # use functions
    print(predict_image(image_path, cfg_path, labels_path, checkpoint_path))
