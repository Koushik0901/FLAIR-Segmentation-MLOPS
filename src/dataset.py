import os
import numpy as np
import pandas as pd
import yaml
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import albumentations as A


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


train_transform = A.Compose(
    [
        A.ChannelDropout(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
    ]
)



class BrainMRIDataset(Dataset):
    def __init__(self, df, transform=None, mean=0.5, std=0.25):
        super(BrainMRIDataset, self).__init__()
        self.df = df
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, raw=False):
        row = self.df.iloc[idx]
        img = cv2.imread(row["image_filename"], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(row["mask_filename"], cv2.IMREAD_GRAYSCALE)
        if raw:
            return img, mask

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        img = transforms.functional.to_tensor(img)
        mask = mask // 255
        mask = torch.Tensor(mask)
        return img, mask


def get_loader(config):
    train_df = pd.read_csv(config["dataset"]["train_csv"])
    val_df = pd.read_csv(config["dataset"]["valid_csv"])
    test_df = pd.read_csv(config["dataset"]["test_csv"])

    train_dataset = BrainMRIDataset(train_df, train_transform)
    val_dataset = BrainMRIDataset(val_df)
    test_dataset = BrainMRIDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loader(config)
    for x, y in train_loader:
        print(x.shape)
        print(y.shape)
        break
