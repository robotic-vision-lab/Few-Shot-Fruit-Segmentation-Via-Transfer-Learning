import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image


class SemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, joint_transform=None, transform=None, target_transform=None):
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.root_dir = root_dir
        
        image_file_names = [f for f in os.listdir(self.root_dir+"/images") if '.png' in f]
        mask_file_names = [f for f in os.listdir(self.root_dir+"/masks") if '.png' in f]
        
        self.images = sorted(image_file_names)
        self.masks = sorted(mask_file_names)
        
        self.id_to_trainid = {}
        self.id_to_trainid[0] = 0
        for i in range(1, 256):
            self.id_to_trainid[i] = 2

        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir+"/images", self.images[idx])).convert('RGB')
        mask = Image.open(os.path.join(self.root_dir+"/masks", self.masks[idx])).convert('L')
        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))
        
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        
        edge = cv2.Canny(np.array(mask), 0.1, 0.2)
        kernel = np.ones((4, 4), np.uint8)
        edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask, edge