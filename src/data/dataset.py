import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class GIDDataset(Dataset):
    """
    GID Dataset için özelleştirilmiş, modüler ve güvenli veri yükleyici.
    Cavit'in imdecode mantığı ile güçlendirilmiştir.
    """
    def __init__(self, images_dir, masks_dir, transform=None, limit=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Dosya isimlerini eşleştir (Uzantıları dikkate almadan ID bazlı)
        img_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if not f.startswith('.')}
        msk_files = {os.path.splitext(f)[0]: f for f in os.listdir(masks_dir) if not f.startswith('.')}
        
        self.ids = sorted(list(set(img_files.keys()) & set(msk_files.keys())))
        
        if limit:
            self.ids = self.ids[:limit]
            
        self.img_names = [img_files[i] for i in self.ids]
        self.msk_names = [msk_files[i] for i in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.img_names[idx])
        msk_path = os.path.join(self.masks_dir, self.msk_names[idx])

        # 1. Görüntüyü Oku (Cavit'in Unicode destekli imdecode yöntemi)
        img_bytes = np.fromfile(img_path, np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Maskeyi Oku ve Yeşil Alanı Ayıkla ([0, 255, 0])
        msk_bytes = np.fromfile(msk_path, np.uint8)
        mask_raw = cv2.imdecode(msk_bytes, cv2.IMREAD_COLOR)
        mask_raw = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2RGB)
        
        # Sadece yeşil kanalları 1, diğerlerini 0 yap
        mask = np.all(mask_raw == [0, 255, 0], axis=-1).astype(np.float32)

        # 3. Augmentation (Albumentations kütüphanesi için hazır yapı)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Manuel Tensor dönüşümü (Eğer transform yoksa)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask
