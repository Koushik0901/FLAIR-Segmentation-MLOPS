import os
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


train_transform = A.Compose(
    [
        A.Resize(
            height=config["dataset"]["image_height"],
            width=config["dataset"]["image_width"],
        ),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1.0),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(
            mean=config["dataset"]["mean"],
            std=config["dataset"]["std"],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(
            height=config["dataset"]["image_height"],
            width=config["dataset"]["image_width"],
        ),
        A.Normalize(
            mean=config["dataset"]["mean"],
            std=config["dataset"]["std"],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)


class BrainMRIDataset(Dataset):
    def __init__(self, csv_file: str, transform=None) -> None:
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        mask_path = self.df.iloc[idx]["mask_path"]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask.unsqueeze(0)


def get_loader(config):
    train_dataset = BrainMRIDataset(config["dataset"]["train_csv"], train_transform)
    val_dataset = BrainMRIDataset(config["dataset"]["eval_csv"], val_transforms)

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

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_loader(config)
    for x, y in train_loader:
        print(x.shape)
        print(y.shape)
        break