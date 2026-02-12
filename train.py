# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from src.dataset import GIDSegmentationDataset
from src.utils import train_one_epoch, validate, visualize_step

# --- AYARLAR ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = '/content/drive/MyDrive/Dataset_Final/' 
TARGET_COLOR_RGB = [0, 255, 0]
BATCH_SIZE = 4
LIMIT = 32 

# --- DATA LOADER ---
train_dataset = GIDSegmentationDataset(base_dir=BASE_PATH, split='train', target_color=TARGET_COLOR_RGB, limit=LIMIT)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = GIDSegmentationDataset(base_dir=BASE_PATH, split='val', limit=LIMIT)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- MODEL ---
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
    def forward(self, x):
        return self.conv(x)

model = DummyModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- ÇALIŞTIRMA ---
if __name__ == "__main__":
    print(f"Eğitim başlıyor... Cihaz: {device}")
    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"İlk epoch tamamlandı. Loss: {loss:.4f}")
    # Not: Görselleştirme terminalde hata verebilir ama Colab'da çalışır
    # visualize_step(model, val_loader, device)