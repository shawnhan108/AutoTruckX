import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import img_dir, csv_src

class TruckDataset(Dataset):
    """
    # (1) must cast from np to tensor: torch.from_numpy(x).permute(2, 0, 1) # D, H, W
    # (2) transformations: resize, totensor, normalize, augmentation?
    # (3) must return leftImg, centerImg, rightImg, leftAng, centerAng, rightAng, all of shape N x Img
    """
    def __init__(self, X, y, model_name):
        self.img_names = X
        self.angles = y
        self.model_name = model_name

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        name = self.img_names[index][0].split('IMG/center')[1]
        front_name, left_name, right_name = os.path.join('data', 'IMG/center' + name), os.path.join('data', 'IMG/left' + name), os.path.join('data', 'IMG/right' + name)
        front_img, left_img, right_img = np.array(Image.open(front_name)), np.array(Image.open(left_name)), np.array(Image.open(right_name))
        front_angle, left_angle, right_angle = self.angles[index][0], self.angles[index][0] + 0.4, self.angles[index][0] - 0.4

        front_img, front_angle = self.process(front_img, front_angle, self.model_name)
        left_img, left_angle = self.process(left_img, left_angle, self.model_name)
        right_img, right_angle = self.process(right_img, right_angle, self.model_name)

        return left_img, front_img, right_img, left_angle, front_angle, right_angle

    def process(self, img, angle, model_name):

        img = torch.from_numpy(img).permute(2, 0, 1) # D, H, W

        if model_name == "TruckNN":
            size = (80, 240)
        elif model_name == "TruckInception":
            size = (299, 299)

        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Lambda(lambda x: (x / 127.5) - 1),
        ])

        img = transform(img)

        # augmentation
        if np.random.rand() < 0.4:
            flip_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=1)
            ])
            img = flip_transform(img)
            angle = angle * -1.0
        
        return img, angle
