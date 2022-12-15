import glob
import datetime

import numpy as np
import pandas as pd
import yaml
import torch
from efficientnet_pytorch import EfficientNet
from torch import nn
from torch.nn import functional as F
from addict import Dict
from train import Model
import argparse
import sys
import tqdm
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
from data import create_dataloader, create_transforms, create_dataset
import ttach as tta

from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A

from captum.influence import TracInCP, TracInCPFast, TracInCPFastRandProj
from captum.influence._utils.common import _load_flexible_state_dict


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='yaml config path')



IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    ten = x.clone()
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    return (ten.permute(1, 2, 0) * 255).numpy().astype(int)

def inverse_normalize(image):
    image = denormalize(image)
    return image

imshow_transform = lambda tensor_in_dataset: inverse_normalize(tensor_in_dataset.squeeze().cpu())


def display_test_example(example, true_label, predicted_label, predicted_prob, label_to_class, name):
    fig, ax = plt.subplots()
    label = 'true/pred: ' + str(label_to_class[true_label]) + ' / ' + str(label_to_class[predicted_label]) + \
    ' predicted_prob: ' + str(predicted_prob.item())
    example = imshow_transform(example)
    ax.imshow(example)
    ax.set_title(label)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    plt.show()
    plt.savefig(f'/kaggle/working/results/test_{name}.png')


def display_training_examples(examples, true_labels, label_to_class, name, figsize=(10, 4)):
    fig = plt.figure(figsize=figsize)
    num_examples = len(examples)
    for i in range(num_examples):
        ax = fig.add_subplot(1, num_examples, i + 1)
        example = examples[i]
        example = imshow_transform(example)
        ax.imshow(example)
        ax.set_title(label_to_class[true_labels[i]])
    plt.show()
    plt.savefig(f'/kaggle/working/results/training_{name}.png')
    return fig


def display_proponents_and_opponents(correct_dataset, label_to_class, test_examples_batch, proponents_indices, opponents_indices,
                                     test_examples_true_labels, test_examples_predicted_labels,
                                     test_examples_predicted_probs):

    for i, (
            test_example,
            test_example_proponents,
            test_example_opponents,
            test_example_true_label,
            test_example_predicted_label,
            test_example_predicted_prob,
    ) in enumerate(zip(
        test_examples_batch,
        proponents_indices,
        opponents_indices,
        test_examples_true_labels,
        test_examples_predicted_labels,
        test_examples_predicted_probs,
    )):
        display_test_example(
            test_example,
            test_example_true_label,
            test_example_predicted_label,
            test_example_predicted_prob,
            label_to_class,
            name=str(i),
        )

        test_example_proponents_tensors, test_example_proponents_labels = zip(
            *[correct_dataset[i] for i in test_example_proponents]
        )
        display_training_examples(
            test_example_proponents_tensors, test_example_proponents_labels, label_to_class, name=str(i) + 'prop', figsize=(20, 8)
        )

        test_example_opponents_tensors, test_example_opponents_labels = zip(
            *[correct_dataset[i] for i in test_example_opponents]
        )
        display_training_examples(
            test_example_opponents_tensors, test_example_opponents_labels, label_to_class, name=str(i) + 'opp', figsize=(20, 8)
        )


def checkpoints_load_func(net, path):
    ckpt = torch.load(path, map_location=DEVICE)
    net.load_state_dict(ckpt, strict=False)
    return 1.


class MyModule(nn.Module):

    def __init__(self, num_classes=25, architecture='efficientnet-b0'):
        super().__init__()
        self.net = EfficientNet.from_pretrained(architecture)
        self.net._fc = nn.Linear(in_features=self.net._fc.in_features, out_features=num_classes, bias=True)

    def forward(self, input):
        return self.net.forward(input)


def test(opt_parser):

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    with open(opt_parser.config, 'r') as cfg:
        opt = Dict(yaml.load(cfg, Loader=yaml.FullLoader))

    best_checkpoint_path = os.path.join('/kaggle/working/best.ckpt')
    correct_dataset_checkpoint_paths = glob.glob(os.path.join('/kaggle/working/checkpoints', "*.ckpt"))

    print('Use best checkpiont: ', best_checkpoint_path)
    print('Found checkpoints: ', len(correct_dataset_checkpoint_paths))
    print(DEVICE)
    with open(opt.labelmap_path, 'r') as f:
        labels_names = f.readlines()
    df = pd.read_csv(opt.df_path, index_col=0)
    test_df = df[df['fold'] == 0]
    train_df = df[df['fold'] != 0].sample(13104)  # TO divide by 112
    print(f'Test count {len(test_df)} images.')
    print(f'Train count {len(train_df)} images.')

    transforms = create_transforms(opt, mode='val')

    correct_dataset = create_dataset(train_df, opt.data_root, transforms=transforms, device=DEVICE)
    test_dataset = create_dataset(test_df, opt.data_root, transforms=transforms, device=DEVICE)

    # net = MyModule()
    # ckpt = torch.load(best_checkpoint_path, map_location=torch.device('cpu'))
    # net.load_state_dict(ckpt['state_dict'], strict=False)

    net = EfficientNet.from_pretrained('efficientnet-b0')
    net._fc = nn.Linear(in_features=net._fc.in_features, out_features=25, bias=True)
    checkpoints_load_func(net, best_checkpoint_path)
    net.eval()
    net.to(DEVICE)

    test_examples_indices = [1000, 1200, 2345, 2700, 3000]
    test_examples_batch = torch.stack([test_dataset[i][0] for i in test_examples_indices])
    test_examples_predicted_probs, test_examples_predicted_labels = torch.max(F.softmax(net(test_examples_batch.to(DEVICE)), dim=1), dim=1)
    test_examples_true_labels = torch.Tensor([test_dataset[i][1] for i in test_examples_indices]).long().to(DEVICE)

    print('Params: ', len(test_examples_batch), len(test_examples_predicted_probs), len(test_examples_true_labels))


    # fig, ax = plt.subplots()
    # example = imshow_transform(test_examples_batch[0])
    # # example = np.clip(example, 0, 1)
    # ax.imshow(example)
    # plt.show()
    # plt.savefig('/kaggle/working/test_example_before.png')


    tracin_cp_fast = TracInCPFast(
        model=net,
        final_fc_layer=list(net.children())[-1],
        influence_src_dataset=correct_dataset,
        checkpoints=correct_dataset_checkpoint_paths,
        checkpoints_load_func=checkpoints_load_func,
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),  # TODO class weights
        batch_size=128,  # Магическое число 112 для influence
        vectorize=False,
    )

    # Можно запускать для поиска оппонентов и пропонентов
    k = 5
    start_time = datetime.datetime.now()
    proponents_indices, proponents_influence_scores = tracin_cp_fast.influence(
        test_examples_batch, test_examples_true_labels, k=k, proponents=True
    )
    opponents_indices, opponents_influence_scores = tracin_cp_fast.influence(
        test_examples_batch, test_examples_true_labels, k=k, proponents=False
    )
    total_minutes = (datetime.datetime.now() - start_time).total_seconds() / 60.0
    print(
        "Computed proponents / opponents over a dataset of %d examples in %.2f minutes"
        % (len(correct_dataset), total_minutes)
    )

    display_proponents_and_opponents(
        correct_dataset,
        labels_names,
        test_examples_batch,
        proponents_indices,
        opponents_indices,
        test_examples_true_labels,
        test_examples_predicted_labels,
        test_examples_predicted_probs,
    )

    # Поиск плохой разметки
    #start_time = datetime.datetime.now()
    #self_influence_scores = tracin_cp_fast.influence()
    #total_minutes = (datetime.datetime.now() - start_time).total_seconds() / 60.0
    #print('computed self influence scores for %d examples in %.2f minutes' % (len(self_influence_scores), total_minutes))


if __name__ == "__main__":

    opt_parser = parser.parse_args()

    test(opt_parser)
