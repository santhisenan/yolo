from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PennFudanDataset(Dataset):
    """Torch dataset for PennFudan Pedestrians dataset.
    Reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

    Args:
        root (str): Path to folder containing dataset
        transforms (): Transforms
    """

    def __init__(self, root, transforms):
        self.root = Path(root)
        self.transforms = transforms

        self.images_folder = self.root / "PNGImages"
        self.masks_folder = self.root / "PedMasks"

        # load all image files, sorting them to
        # ensure that they are aligned
        self.img_paths = list(sorted(self.images_folder.glob("*.png")))
        self.mask_paths = list(sorted(self.masks_folder.glob("*.png")))

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)

        obj_ids = np.unique(mask)[1:]
        # split one mask with all instances into multiple binary masks
        masks = mask == obj_ids[:, None, None]

        num_objects = len(obj_ids)

        boxes = []
        for m in masks:
            pos = np.where(m)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes)

        # all labels are 1
        labels = torch.ones((num_objects,), dtype=torch.int64)

        target = {}
        target["image_id"] = 0
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.img_paths)


def get_transform(train):
    tfms = []
    tfms.append(transforms.ToTensor())
    tfms.append(transforms.Resize((448, 448)))

    if train:
        tfms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(tfms)


def get_dataloader(train=True):
    dataset = PennFudanDataset(
        root="../../data/PennFudanPed", transforms=get_transform(train=train)
    )

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    return data_loader


def collate_fn(batch):
    """Taken from https://github.com/pytorch/vision/blob/27745e5be086ecdf2a8e609d65c13969cb5201f7/references/detection/utils.py#L203"""
    return tuple(zip(*batch))
