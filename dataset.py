import os
import cv2
import torch
import albumentations as A

import config as CFG


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, locations, transforms):
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.locations = locations
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {}

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['location'] = self.locations[idx]

        return item


    def __len__(self):
        return len(self.locations)

def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

    