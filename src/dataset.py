# src/dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class GIDSegmentationDataset(Dataset):
    def __init__(self, base_dir, split='train', target_color=[0, 255, 0], limit=None):
        self.base_dir = base_dir
        self.target_color = np.array(target_color)
        self.image_dir = os.path.join(base_dir, 'Normal', split)
        self.mask_dir = os.path.join(base_dir, 'Mask', split)
        
        # Klasör kontrolü ve dosya listeleme
        if os.path.exists(self.image_dir):
            self.images = sorted(os.listdir(self.image_dir))
            self.masks = sorted(os.listdir(self.mask_dir))
        else:
            self.images = []
            self.masks = []
            print(f"Uyarı: Yol bulunamadı -> {self.image_dir}")

        if limit and len(self.images) > 0:
            self.images = self.images[:limit]
            self.masks = self.masks[:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Dosya yolları
        img_name = self.images[index]
        mask_name = self.masks[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Yükleme ve Dönüştürme
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        
        mask_binary = np.all(mask == self.target_color, axis=-1).astype(np.float32)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        return image_tensor, mask_tensor