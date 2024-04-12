import os
import cv2
import torch
import albumentations as A
import json
import re
from PIL import Image
from torchvision import transforms
import config as CFG
import pdb

class CPIPDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the image filenames and locations.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df

    def __getitem__(self, idx):
        # Retrieve image filename and location from the DataFrame
        image_path = self.df.iloc[idx]['image']
        location = self.df.iloc[idx]['location']

        # Prepare image
        image = prepare_image(image_path)

        # Convert location to tensor
        location_tensor = torch.tensor(location, dtype=torch.float)

        item = {'image': image, 'location': location_tensor}

        return item
    
    def __len__(self):
        return len(self.df)

def prepare_image(image_path, transforms=get_transforms()):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transformations if any
    if transforms:
        image = transforms(image=image)['image']

    # Convert image to tensor
    image_tensor = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)

    return image_tensor

def get_transforms():
    return A.Compose(
        [
            A.CenterCrop(CFG.img_height, CFG.img_width, always_apply=True),
            A.Resize(CFG.target_img_height, CFG.target_img_width, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )