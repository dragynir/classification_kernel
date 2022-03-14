import yaml
import glob
import os
import pandas as pd
import torch
from torch.nn import functional as F
from addict import Dict
from data import create_dataset, create_dataloader, create_transforms
from train import Model
import ttach as tta
import torch.nn as nn
import argparse
import tqdm
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Merger:

    def __init__(
            self,
            type: str = 'mean',
            n: int = 1,
    ):

        if type not in ['mean', 'gmean', 'tsharpen']:
            raise ValueError('Not correct merge type `{}`.'.format(type))

        self.output = None
        self.type = type
        self.n = n
        self.all_outputs = []

    def append(self, x):

        if self.type == 'tsharpen':
            x = x ** 0.5

        if self.output is None:
            self.output = x
        elif self.type in ['mean', 'tsharpen']:
            self.output = self.output + x
        elif self.type == 'gmean':
            self.output = self.output * x

        self.all_outputs.append(F.softmax(x, dim=1))

    @property
    def result(self):

        if self.type in ['mean', 'tsharpen']:
            result = self.output / self.n
        elif self.type in ['gmean']:
            result = self.output ** (1 / self.n)
        else:
            raise ValueError('Not correct merge type `{}`.'.format(self.type))

        all_results = torch.stack(self.all_outputs, dim=1)

        return F.softmax(result, dim=1), all_results


class ClassificationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (classification model) with test time augmentation transforms
    Args:
        model (torch.nn.Module): classification model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_label_key (str): if model output is `dict`, specify which key belong to `label`
    """

    def __init__(
        self,
        model,
        transforms,
        merge_mode = "mean",
        output_label_key = None,
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_label_key

    def forward(
        self, image: torch.Tensor, *args
    ):
        merger = Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_label(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result


def predict_loader(model, dataloader, top_k, tta):

    y_pred = []
    y_pred_conf = []

    with torch.no_grad():
        for imgs, imgs_name in tqdm.tqdm(dataloader):

            # print(imgs_name)

            if tta:
                probs, all_probs = model(imgs.to(DEVICE))
            else:
                logits = model(imgs.to(DEVICE))
                probs = F.softmax(logits, dim=1)

            confidence, top_labels = torch.topk(probs, top_k, dim=1)
            pred_labels = top_labels.cpu().detach().numpy()
            pred_confidence = confidence.cpu().detach().numpy()
            # print(pred_labels.shape)
            # print(pred_confidence.shape)
            if tta:
                top_confidence = torch.index_select(all_probs, dim=-1, index=top_labels.squeeze())
                confidence_max, _ = torch.max(top_confidence, dim=1)
                # print('Max conf:', torch.round(confidence_max * (10**3)) / (10**3))

            y_pred.extend(pred_labels)
            y_pred_conf.extend(pred_confidence)
            # print('List shape', y_pred[0].shape)
            # print('Confidence:', torch.round(confidence * (10**3)) / (10**3))

    return y_pred, y_pred_conf

def indexes2labels(labels, indexes):
    return np.array(list(map(lambda x: labels[x], indexes)))

def predict(opt, images_path, model_ckpt, use_tta=False, top_k=5):

    with open(opt.labelmap_path, 'r') as f:
        labels = f.read().splitlines()

    images_links = os.listdir(images_path)
    df = pd.DataFrame({'ids': images_links, 'target': images_links})

    transforms = create_transforms(opt, 'val')

    dataset = create_dataset(df, images_path, transforms)

    dataloader = create_dataloader(dataset, batch_size=32,
                        num_workers=2, shuffle=False, drop_last=False)

    inference_model = Model.load_from_checkpoint(model_ckpt,
        train_dataset=dataset, val_dataset=dataset, opt=opt, strict=False)

    inference_model.eval()
    inference_model.to(DEVICE)

    if use_tta:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
            ]
        )

        inference_model = ClassificationTTAWrapper(inference_model, transforms, merge_mode='mean')

    y_pred, y_pred_conf = predict_loader(inference_model, dataloader, top_k, use_tta)

    pred_labels = list(map(lambda x: indexes2labels(labels, x), y_pred))

    pred_labels = np.vstack(pred_labels)
    y_pred_conf = np.vstack(y_pred_conf)

    res_df = pd.DataFrame({'images': images_links})

    for i in range(top_k):
        res_df[f'top_{i+1}'] = pred_labels[:, i]
        res_df[f'top_{i+1}_conf'] = y_pred_conf[:, i]

    return res_df


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='yaml config path')
parser.add_argument('--images_path', type=str, default=None, help='images folder path')
parser.add_argument('--model_path', type=str, default=None, help='model pt path')
parser.add_argument('--out_df', type=str, default='./predict.csv', help='result df path')
parser.add_argument('--use_tta', action='store_true', help='enable tta')



if __name__ == '__main__':

    opt = parser.parse_args()

    with open(opt.config, 'r') as cfg:
        opt_config = Dict(yaml.load(cfg, Loader=yaml.FullLoader))

    df_res = predict(opt_config, opt.images_path, opt.model_path, opt.use_tta, top_k=5)

    df_res.to_csv(opt.out_df, index=False);
