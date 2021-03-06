import cv2
import os
import pandas as pd
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from albumentations.pytorch.transforms import ToTensorV2
from .transforms import TTA_5_cropps
from utils import calculate_sample_weights

class ImageDataset(Dataset):
    '''
        Return images one by one with augmentations
    '''
    def __init__(self, data_root, df, transforms=None, domain_transforms=None):
        super().__init__()
        self.df = df
        self.data_root = data_root
        self.image_ids = df.ids.values
        self.labels = df.target.values if 'target' in df.columns else None 
        self.images_meta = df.images_meta.apply(json.loads).values if 'images_meta' in df.columns else None
        self.transforms = transforms
        self.domain_transforms = domain_transforms
        self.to_tensor = ToTensorV2()        

    def __crop_image(self, idx, image):
        if self.images_meta is None:
            return image

        image_meta = self.images_meta[idx]

        if 'bbox' in image_meta:
            bbox = image_meta['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            image = image[y:y+h, x:x+w, :]

        return image
    
    def __preprocess_image(self, idx, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.__crop_image(idx, image)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.data_root, image_id)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = self.__preprocess_image(idx, image)

        label = self.labels[idx]

        if not (type(image) is np.ndarray):
            raise ValueError(f'Image is corrupted: {image_path}')

        if self.transforms:
            sample = self.transforms(image=image)
            image  = sample['image']

        if self.domain_transforms:
            image = self.domain_transforms(image, label)

        image = self.to_tensor(image=image)['image']
        if self.labels is None:
            return image

        return image, label

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def get_labels(self) -> list:
        if self.labels is None:
            return None
        return list(self.labels)


class MultiImageDataset(ImageDataset):
    '''
        Return images one by one with augmentations
    '''
    def __init__(self, resize_size, target_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize_size, self.target_size = resize_size, target_size

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.data_root, image_id)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = self.__preprocess_image(idx, image)

        label = self.labels[idx]

        if not (type(image) is np.ndarray):
            raise ValueError(f'Image is corrupted: {image_path}')

        if self.transforms:
            sample = self.transforms(image=image)
            image  = sample['image']

        if self.domain_transforms:
            image = self.domain_transforms(image, label)

        images = TTA_5_cropps(image, self.resize_size, self.target_size)
        tensors = []
        for img in images:
            img = self.to_tensor(image=img)['image']
            img = torch.unsqueeze(img, dim=0)
            tensors.append(img)
        
        multi_image = torch.cat(tensors, dim=0)

        if self.labels is None:
            return multi_image

        return multi_image, label


def create_dataset(df: pd.DataFrame, data_root, transforms=None, domain_transforms=None) -> ImageDataset:
    '''
        Create dataset from data.csv DataFrame
    '''
    
    return ImageDataset(data_root, df, transforms, domain_transforms)

def create_multi_input_dataset(
    df: pd.DataFrame,
    data_root,
    resize_size,
    target_size,
    transforms=None,
    domain_transforms=None
   ) -> ImageDataset:
    '''
        Create dataset from data.csv DataFrame
    '''
    

    return MultiImageDataset(
        resize_size,
        target_size,
        data_root=data_root,
        df=df,
        transforms=transforms,
        domain_transforms=domain_transforms)


def create_dataloader(dataset: ImageDataset, batch_size, num_workers, shuffle, sample_weights=False, drop_last=True):

    sampler = None
    if sample_weights:
        weights = calculate_sample_weights(dataset.df, over='target', beta=0.999)
        print('Calculated sample weights:', pd.Series(weights).value_counts())
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
        shuffle = False

    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      pin_memory=True,
                      sampler=sampler,
                      )
