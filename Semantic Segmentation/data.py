import os

import torch 
from torch.utils.data import Dataset 
from torchvision import transforms

from PIL import Image 
import numpy as np

from utils import split_img


class CityscapeDataset(Dataset):
    def __init__(self, img_dir, img_dim, mode, cluster_model):
        # mode is either train or valid
        # cluster_model: the KMeans model for clustering img to class labels
        self.img_dir = os.path.join(img_dir, mode)
        self.img_dim = img_dim
        self.img_names = os.listdir(self.img_dir)
        self.cluster_model = cluster_model
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        pil_img = Image.open(os.path.join(self.img_dir, self.img_names[idx])).convert('RGB')
        orig_img, mask = split_img(np.array(pil_img), self.img_dim)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])
        orig_img = transform(orig_img)

        class_map = self.cluster_model.predict(mask.reshape(-1, 3)).reshape(self.img_dim, self.img_dim)
        class_map = torch.Tensor(class_map).long()

        return orig_img, class_map
    
