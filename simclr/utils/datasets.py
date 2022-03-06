import cv2
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class SimclrDataset(Dataset):
    def __init__(self, data_root, image_ids, labels, transforms):
        super().__init__()
        self.data_root = data_root
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.data_root, image_id)
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = torch.tensor(image)
        #
        return self.transforms(image), label

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def get_labels(self) -> list:
        if self.labels is None:
            return None
        return list(self.labels)

def create_dataset(df: pd.DataFrame, data_root, transforms):

    image_ids = df.ids.values
    labels = None

    if 'target' in df.columns:
        labels = df.target.values

    return SimclrDataset(data_root, image_ids, labels, transforms)