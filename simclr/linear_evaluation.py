import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import random

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR

from utils import yaml_config_hook
from utils import create_dataset
from utils import visualize_embeddings
from sklearn.metrics import f1_score, classification_report


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(args, loader, simclr_model, model, criterion, metric, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    metric_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)


        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc
        metric_epoch+= metric(y.cpu(), predicted.cpu())


        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch, metric_epoch


def test(args, loader, simclr_model, model, criterion, metric, names):
    loss_epoch = 0
    accuracy_epoch = 0
    metric_epoch = 0
    model.eval()

    true_labels = []
    pred_labels = []

    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        true_labels.extend(y.numpy())

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc
        metric_epoch+= metric(y.cpu(), predicted.cpu())

        pred_labels.extend(predicted.cpu().numpy())

        loss_epoch += loss.item()
    print(labels)
    print(classification_report(true_labels, pred_labels, target_names=names))

    return loss_epoch, accuracy_epoch, metric_epoch

def extract_vall_loss(name):
    return float(name.split('=')[-1].replace('.ckpt', ''))

def get_best_checkpoint(checkpoint_path):

    checkpoints = os.listdir(checkpoint_path)
    checkpoints = list(filter(lambda x: 'tmp' not in x, checkpoints))

    best_ckpt = np.argmin(map(extract_vall_loss, checkpoints))

    return os.path.join(checkpoint_path, checkpoints[best_ckpt])


if __name__ == "__main__":

    seed_everything(42)

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    labels = None

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == 'custom':
        df_folds = pd.read_csv(args.df_path, index_col=0)

        with open(args.labelmap_path, 'r') as f:
            classes = f.read().splitlines()
            df_folds = df_folds[df_folds['class'].isin(classes)].reset_index(drop=True)

        # image per class
        def images_per_class(df, msg):
            class_stats_group = df.groupby('class')['ids'].count()
            class_stats_df = pd.DataFrame(class_stats_group.to_dict().items())
            class_stats_df.columns = ['class', 'count']
            print(f'{msg} tatistic: \n', class_stats_df)


        unique_labels = df_folds['target'].unique()

        mapping = {k:v for k, v in zip(unique_labels, range(len(unique_labels)))}

        df_folds['target'] = df_folds['target'].map(mapping)

        class_mapping = df_folds.groupby('target').first().reset_index()[['target', 'class']].sort_values(by='target')
        labels = class_mapping['class'].values


        print('Class mapping: \n', class_mapping)

        print('Use classes: ', df_folds['class'].unique())
        print('Use targets: ', df_folds['target'].unique())

        train_df = df_folds[df_folds['fold'] != 0]
        images_per_class(train_df, 'Train distribution\n')
        train_dataset = create_dataset(train_df, args.data_root, transforms=TransformsSimCLR(size=args.image_size).test_transform)

        test_df = df_folds[df_folds['fold'] == 0]
        images_per_class(test_df, 'Test distribution\n')
        test_dataset = create_dataset(test_df, args.data_root, transforms=TransformsSimCLR(size=args.image_size).test_transform)

    else:
        raise NotImplementedError


    ## Logistic Regression
    n_classes = 10  # CIFAR-10 / STL-10

    if args.dataset == 'custom':
        n_classes = len(train_df.target.unique())

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)


    # model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    # simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))

    checkpoint_path = get_best_checkpoint(os.path.join(args.experiment_path, 'checkpoint'))

    state_dict = torch.load(checkpoint_path, map_location=args.device)['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '')] = state_dict.pop(key)

    simclr_model.load_state_dict(state_dict)

    simclr_model.eval()

    simclr_model.to(args.device)


    model = LogisticRegression(
            simclr_model.n_features,
            n_classes,
            args.logistic_projection,
            args.logistic_dropout
            )

    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        simclr_model, train_loader, test_loader, args.device
    )

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logistic_batch_size
    )

    # enable tsne visualization of embeddings
    # visualize_embeddings(
    #     test_X,
    #     test_y,
    #     os.path.join(args.experiment_path, 'tsne_plot_test.png'),
    #     n_classes
    # )

    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch, metric_epoch = train(
            args, arr_train_loader, simclr_model, model, criterion, f1_score, optimizer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t"
            f"Accuracy: {accuracy_epoch / len(arr_train_loader)} \t"
            f"Fscore: {metric_epoch  / len(arr_train_loader)} \t"
        )

    # final testing
    loss_epoch, accuracy_epoch, metric_epoch = test(
        args, arr_test_loader, simclr_model, model, criterion, f1_score, labels
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)} \t"
        f"Fscore: {metric_epoch / len(arr_test_loader)}"
    )
