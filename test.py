import numpy as np
import pandas as pd
import yaml
import torch
from torch.nn import functional as F
from addict import Dict
from train import Model
import argparse
import sys

import shap
import cv2
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, classification_report
from data import create_dataloader, create_transforms
import ttach as tta

from albumentations.pytorch.transforms import ToTensorV2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestDataset(Dataset):
    def __init__(self, data_root, image_ids, labels, resolution, transforms=None):
        super().__init__()
        self.data_root = data_root
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms
        self.resolution = resolution
        self.to_tensor = ToTensorV2()

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]


        image = cv2.imread(os.path.join(self.data_root, image_id), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not (type(image) is np.ndarray):
            raise TypeError('Image is None')

        tr_image = image

        if self.transforms:
            sample = self.transforms(image=image)
            tr_image  = sample['image']

        tr_image = self.to_tensor(image=tr_image)['image']
        label = self.labels[idx]

        return cv2.resize(image, (self.resolution, self.resolution)), tr_image, label, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def get_labels(self) -> list:
        return list(self.labels)

def create_test_dataset(df: pd.DataFrame, data_root, resolution, transforms=None) -> TestDataset:

    image_ids = df.ids.values
    labels = df.target.values

    return TestDataset(data_root, image_ids, labels, resolution, transforms)


def evaluate(model, loader, output_path, labels_names, write_batches_count, show_shap_values, transforms):
    y_true = []
    y_pred = []

    explainer = None
    if show_shap_values:
        def f(batch):
            with torch.no_grad():
                processed_imgs = []
                for img in batch:
                    sample = transforms(image=img)
                    processed_imgs.append(sample['image'])
                pr_batch = torch.stack(processed_imgs)
                tmp = pr_batch.to(DEVICE)
                out = model(tmp).cpu()
            return out

        imgs, _, _, _ = next(iter(loader))
        masker = shap.maskers.Image("inpaint_telea", imgs.shape[1:])
        explainer = shap.Explainer(f, masker)


    count = 0
    for imgs, tr_imgs, labels, images_ids in loader:

        with torch.no_grad():
            logits = model(tr_imgs.to(DEVICE))
            probs = F.softmax(logits, dim=1)
            pred_labels = torch.argmax(probs, dim=1).cpu().detach().numpy()

        # save less then write_batches_count images examples
        if count < write_batches_count:

            shap_values = None
            if explainer:
                exp_input = imgs.detach().cpu().numpy().astype(np.float32)
                shap_values = explainer(exp_input, max_evals=1000,
                    batch_size=64, outputs=shap.Explanation.argsort.flip[:1])

            for i, (img, pred_lb, true_lb, img_id) in enumerate(zip(imgs, pred_labels, labels, images_ids)):
                img = img.detach().numpy()
                true_name = labels_names[true_lb].strip()
                pred_name = labels_names[pred_lb].strip()
                label_info = f'{pred_name} / {true_name}'

                if shap_values:
                    shap.image_plot(shap_values[i:i+1], show=False)
                    plt.title(label_info)
                else:
                    plt.figure()
                    plt.imshow(img)
                    plt.title(label_info)

                plt.savefig(os.path.join(output_path, img_id.split('/')[-1]))

        y_true.extend(labels)
        y_pred.extend(pred_labels)

        count+=1

    print(f'Finished {count} batches.')
    return np.array(y_true), np.array(y_pred)


def plot_precision_recall_bar(labels, precision_values, recall_values, save_path, figsize):

    df = pd.DataFrame({
                        'labels': labels + labels,
                        'values': precision_values + recall_values,
                        'type':   ['precision'] * len(precision_values) + \
                                    ['recall'] * len(recall_values)
                        })

    plt.figure(figsize=figsize)
    sns.barplot(x='labels', y='values', hue="type", data=df)
    plt.title('Precision/Recall')
    plt.xticks(rotation=45)
    plt.grid()
    plt.savefig(save_path)



def plot_cm(y_true, y_pred, save_path, figsize=(20,20)):
    '''
        Plot confusion matrix
    '''

    unique_labels = np.unique(np.concatenate([y_true, y_pred]))

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = (cm + 1) / (cm_sum.astype(float) + 1) * 100 # avoid zero division
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    params = {
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'
         }

    plt.rcParams.update(params)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax,
                                    linecolor='black', linewidths=1)

    plt.savefig(save_path)


def run_test(opt, checkpoint_path, output_path, data_root=None, write_batches_count=6,
                fold=0, batch_size=8, test_csv=None, show_shap_values=False, use_tta=False):


    output_images_path = os.path.join(output_path, 'images')
    os.makedirs(output_images_path, exist_ok=True)
    output_stats_path = os.path.join(output_path, 'plots')
    os.makedirs(output_stats_path, exist_ok=True)

    with open(opt.labelmap_path, 'r') as f:
        labels_names = f.readlines()

    if test_csv:
        test_df = pd.read_csv(test_csv, index_col=0)
    else:
        test_df = pd.read_csv(opt.df_path, index_col=0)
        test_df = test_df[test_df['fold'] == fold]

    print(f'Testing on {len(test_df)} images.')

    if data_root:
        opt.data_root = data_root

    transforms = create_transforms(opt, mode='val')
    test_dataset = create_test_dataset(test_df, opt.data_root,
            opt.resolution, transforms=transforms)

    inference_model = Model.load_from_checkpoint(checkpoint_path,
        train_dataset=test_dataset, val_dataset=test_dataset, opt=opt, strict=False)

    inference_model.eval()

    inference_model.to(DEVICE)

    if use_tta:
        transforms = tta.Compose(
            [
                #tta.Rotate90(angles=[0, 90]),
                tta.HorizontalFlip(),
                # tta.FiveCrops(200, 200)
            ]
        )

        inference_model = tta.ClassificationTTAWrapper(inference_model, transforms, merge_mode='mean')


    test_loader = create_dataloader(test_dataset, batch_size=batch_size,
                            num_workers=2, shuffle=True, drop_last=False)

    y_true, y_pred = evaluate(inference_model, test_loader,
                            output_images_path, labels_names, write_batches_count, show_shap_values, transforms)

    index_to_name = lambda x: labels_names[x]

    unique_class_index = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    unique_class_names = list(map(index_to_name, unique_class_index))
    print('Unique classes: ', unique_class_names)
    print(y_true.shape, y_pred.shape)
    print(np.unique(y_true, return_counts=True))
    print(np.unique(y_pred, return_counts=True))

    print('Weighted Fscore:', f1_score(y_true, y_pred, average='weighted'))
    print('Fscore:', f1_score(y_true, y_pred, average='macro'))


    print(classification_report(y_true, y_pred, target_names=unique_class_names))

    precision_values = precision_score(y_true, y_pred, average=None)
    recall_values = recall_score(y_true, y_pred, average=None)

    y_true = list(map(index_to_name, y_true))
    y_pred = list(map(index_to_name, y_pred))

    print(f'Total classifications: {len(y_pred)}.')

    plot_cm(y_true, y_pred, os.path.join(output_stats_path, 'conf_matrix.png'), figsize=(75,75))

    plot_precision_recall_bar(unique_class_names, precision_values.tolist(), recall_values.tolist(),
        os.path.join(output_stats_path, 'precision_recall_bar.png'), figsize=(100, 30))


def extract_vall_loss(name):
    return float(name.split('=')[-1].replace('.ckpt', ''))

def get_best_checkpoint(checkpoint_path):

    checkpoints = os.listdir(checkpoint_path)

    # ignore checkpoint with 'tmp' in name
    checkpoints = list(filter(lambda x: 'tmp' not in x, checkpoints))

    best_ckpt = np.argmin(map(extract_vall_loss, checkpoints))

    return os.path.join(checkpoint_path, checkpoints[best_ckpt])

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='yaml config path')
parser.add_argument('--val', action='store_true', help='Evaluate data.csv')
parser.add_argument('--test', action='store_true', help='Evaluate data_test.csv')
parser.add_argument('--fold', type=int, default=0, help='data.csv fold to evaluate')
parser.add_argument('--shap', action='store_true', help='Save images with shap values')
parser.add_argument('--tta', action='store_true', help='Use Test time augmentations')

def test(opt_parser):

    with open(opt_parser.config, 'r') as cfg:
        opt = Dict(yaml.load(cfg, Loader=yaml.FullLoader))

    output_path = os.path.join(opt.experiment_path, 'results')
    os.makedirs(output_path, exist_ok=True)

    checkpoint_path = os.path.join(opt.experiment_path, 'checkpoint')
    best_checkpoint_path = get_best_checkpoint(checkpoint_path)

    print('Use best checkpiont: ', best_checkpoint_path)

    if opt_parser.val:
        output_path = os.path.join(output_path, 'val')
        run_test(opt, best_checkpoint_path, output_path, data_root=opt.data_root,
            write_batches_count=8, fold=opt_parser.fold, batch_size=8, test_csv=None, show_shap_values=opt_parser.shap)
    elif opt_parser.test:
        csv_path = os.path.join(opt.experiment_path, 'dataset', 'data_test.csv')
        output_path = os.path.join(output_path, 'test')
        run_test(opt, best_checkpoint_path, output_path, data_root=opt.test_data_root,
            write_batches_count=8, fold=opt_parser.fold, batch_size=8, test_csv=csv_path,
            show_shap_values=opt_parser.shap, use_tta=opt_parser.tta)
    else:
        print('Please add --val or --test flag')
        return

if __name__ == "__main__":

    opt_parser = parser.parse_args()

    test(opt_parser)
