import glob
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from addict import Dict
import yaml

import imagehash
import torch
from PIL import Image

MERGE_GROUPS = {

}


def create_horizontal_plot(df, title, x_data, y_data, label=None, save_name=None, show_values=True):

    plt.figure(figsize=(18, 20))
    ax = plt.gca()

    df.sort_values(by=x_data, inplace=True, ascending=False)

    sns.set_color_codes("pastel")
    sns.barplot(x=x_data, y=y_data, data=df, color=None, label=None, orient="h", ax=ax)

    # Add a legend and informative axis label
    # ax.legend(ncol=1, loc="lower right", frameon=True)

    ax.set_ylabel("Classes")
    ax.set_xlabel("Count")
    ax.set_title(title, color='#2E86C1')
    sns.despine(left=True, bottom=True)

    if label:
        ax.legend(loc="upper right", frameon=True)

    if show_values:
        for p in ax.patches:
            width = p.get_width()    # get bar length
            ax.text(width + 10,       # set the text at 1 unit right of the bar
                    p.get_y() + p.get_height() / 2, # get Y coordinate + X coordinate / 2
                    int(width), # set variable to display, 2 decimals
                    ha = 'left',   # horizontal alignment
                    va = 'center')  # vertical alignment
    plt.savefig(save_name)


def filter_bad_images(df, data_path):
    '''
        Remove corrupted images from df
    '''
    bad_images_list = []
    for i, row in df.iterrows():
        img_path = os.path.join(data_path, row['ids'])

        if not os.path.exists(img_path):
            bad_images_list.append(row['ids'])
        else:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if not (type(image) is np.ndarray):
                bad_images_list.append(row['ids'])


    if len(bad_images_list) > 0:
        print('Warning! Found corrupted images: ')
        print(bad_images_list)
        # filter None images
        df = df[~df['ids'].isin(bad_images_list)]

    return df



def calulate_hashes(df, data_path):

    funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
    ]


    petids = []
    hashes = []

    for i, row in df.iterrows():

        img_path = os.path.join(data_path, row['ids'])
        image = Image.open(img_path)

        petids.append(row['ids'])
        hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))

    return petids, np.array(hashes)


def drop_duplicates(df, data_path, sim_threshold, iters=4):

    print('Searching for duplicates...')

    petids, hashes_all = calulate_hashes(df, data_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    drop_ids = []
    check_ids = petids
    hashes = hashes_all

    for j in range(iters):
        print(f'Iter {j+1}/{iters}, ', len(check_ids), len(drop_ids))
        hashes_t = torch.Tensor(hashes.astype(int)).to(device)

        sims = np.array([(hashes_t[i] == hashes_t).sum(dim=1).cpu().numpy()/256 for i in range(hashes_t.shape[0])])

        sims = sims * (1 - np.eye(len(sims)))

        indices = np.argwhere(sims > sim_threshold)

        petids1 = [check_ids[ind[0]] for ind in indices]
        petids2 = [check_ids[ind[1]] for ind in indices]

        dups = [tuple(sorted([petid1,petid2])) for petid1, petid2 in zip(petids1, petids2)]

        dup_df = pd.DataFrame({'img1': [d[0] for d in dups], 'img2': [d[1] for d in dups]})

        drop_ids.extend(dup_df['img1'].values.tolist())

        drop_ids = list(set(drop_ids))

        duplicates = np.unique(dup_df['img2'].values)
        check_idx = [check_ids.index(ind) for ind in duplicates]

        hashes = np.array([hashes[i] for i in check_idx])
        check_ids = duplicates.tolist()


    df_clear = df[~df['ids'].isin(drop_ids)]

    droped_count = len(df) - len(df_clear)

    print(f'Dropping {droped_count} duplicates.')

    return df_clear


def check_intersaction(df):
    '''
        Validate data split
    '''
    folds = df['fold'].unique()
    error = False
    for f1 in folds:
        for f2 in folds:

            if f1 == f2:
                continue

            f1_df = df[df['fold'] == f1]
            f2_df = df[df['fold'] == f2]

            inter = set(f1_df['ids'].values) & set(f2_df['ids'].values)

            if len(inter) > 0:
                print('ERROR!')
                print(f'Intersection in: {f1} and {f2} folds')
                print(inter)
                error = True

    return error


def prepare_source(data_path, images_extention, sub_folder=''):

    print(f'Prepare data from {data_path}')
    train_files = []

    if isinstance(images_extention, str):
        train_files = glob.glob(f'{data_path}/**/*.{images_extention}', recursive=True)
    else:
        for ext in images_extention:
            ext_images = glob.glob(f'{data_path}/**/*.{ext}', recursive=True)
            train_files.extend(ext_images)

    df = pd.DataFrame()
    df['ids'] = train_files
    df['ids'] = df['ids'].apply(lambda x: '/'.join(x.split('/')[-2:]))
    df['class'] =  df['ids'].apply(lambda x: x.split('/')[0])
    df['fold'] = 0

    # raw dataset info
    print(f'Raw images count: {len(df)}')
    raw_num_classes = len(df['class'].unique())
    print(f'Raw num classes: {raw_num_classes}')

    # filter None images
    df = filter_bad_images(df, data_path)

    df['ids'] = df['ids'].apply(lambda x: os.path.join(sub_folder, x))

    return df



def merge_classes(df, merge_dict):

    for true_name, fake_names in merge_dict.items():
        for f_name in fake_names:
            df.loc[:, 'class'].replace(f_name, true_name, inplace=True)

    return df


def calculate_stats_plot(df, path):

    class_stats_group = df.groupby('class')['ids'].count()
    class_stats_df = pd.DataFrame(class_stats_group.to_dict().items())
    class_stats_df.columns = ['class', 'count']

    print('Count per classes:', class_stats_group.to_dict())
    num_classes = len(df['target'].unique())

    create_horizontal_plot(class_stats_df,
        f'Images per class for {num_classes} classes', x_data='count', y_data='class', save_name=path)

def prepare_dataset(source_path, data_root, dataset_output_path, classification_group, images_extention='jpg',
                    test_source=False, n_splits=5, seed=42, drop_count=50, sim_threshold=0.90):

    '''
        Create DataFrame dataset from images folders
        filter_labels: filter labels and do not create labels.txt for generated df
    '''

    filter_labels = False
    atrifacts_postfix = ''

    if test_source:
        filter_labels = True
        atrifacts_postfix = '_test'

    csv_path = os.path.join(dataset_output_path, f'data{atrifacts_postfix}.csv')
    labelmap_path = os.path.join(dataset_output_path, 'labelmap.txt')
    plot_path = os.path.join(dataset_output_path, f'barplot{atrifacts_postfix}.png')

    df = None
    if type(source_path) is list:
        df_sources = []
        for data_path in source_path:
            sub_folder = os.path.basename(data_path.strip('/'))
            df_sources.append(prepare_source(data_path, images_extention, sub_folder=sub_folder))
        df = pd.concat(df_sources, ignore_index=True)
    else:
        df = prepare_source(source_path, images_extention)


    df = drop_duplicates(df, data_root, sim_threshold)

    df = merge_classes(df, MERGE_GROUPS[classification_group])

    # drop classes with num_examples <= drop_count
    df = df.groupby('class').filter(lambda x: len(x) > drop_count).reset_index(drop=True)

    if filter_labels:
        with open(labelmap_path, 'r') as f:
            classes = f.read().splitlines()
        df = df[df['class'].isin(classes)].reset_index(drop=True)
        df['target'] = df['class'].apply(lambda x: classes.index(x))

    else:

        classes = sorted(df['class'].unique())
        df['target'] = df['class'].apply(lambda x: classes.index(x))

        with open(labelmap_path, 'w') as f:
            f.write('\n'.join(classes))


    print('Unique classes indexes:', sorted(df['target'].unique()))

    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    for i_fold, (train_index, test_index) in enumerate(skf.split(df, df['target'])):
        df.loc[test_index, 'fold'] = i_fold

    if check_intersaction(df):
        return

    calculate_stats_plot(df, plot_path)

    num_classes = len(df['target'].unique())
    print(f'Writing csv for {num_classes} classes')
    print(f'Total images: {len(df)}')

    df.to_csv(csv_path)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='yaml config path')
parser.add_argument('--test_source', action='store_true', help='Create test dataset')


if __name__ == '__main__':

    opt_parser = parser.parse_args()

    with open(opt_parser.config, 'r') as cfg:
        opt = Dict(yaml.load(cfg, Loader=yaml.FullLoader))

    images_source = opt.images_source
    data_root = opt.data_root
    if opt_parser.test_source:
        images_source = opt.test_images_source
        data_root = opt.test_data_root

    prepare_dataset(images_source, data_root, opt.dataset_path, opt.project_name,
        test_source=opt_parser.test_source, images_extention=['jpg', 'png'], sim_threshold=0.95)
