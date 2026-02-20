import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from config import Config
from src.models.model_factory import get_model
from src.data.dataset import GIDDataset
from src.utils.losses import FocalTverskyLoss
from src.utils.metrics import get_iou_score

def train():
    # 1. Veri Hazırlığı
    train_ds = GIDDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MSK_DIR)
    val_ds = GIDDataset(Config.VAL_IMG_DIR, Config.VAL_MSK_DIR)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 2. Model, Loss ve Optimizer
    model = get_model(Config.MODEL_NAME).to(Config.DEVICE)
    criterion = FocalTverskyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    # FP16 (Hızlı eğitim için Mixed Precision)
    scaler = torch.cuda.amp.GradScaler()

    print(f"🚀 Eğitim Başlıyor: {Config.MODEL_NAME} | Cihaz: {Config.DEVICE}")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss, train_iou = 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{Config.EPOCHS}]")
        for images, masks in loop:
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
            
            # Forward pass with Mixed Precision
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrikler
            train_loss += loss.item()
            train_iou += get_iou_score(outputs, masks)
            
            loop.set_postfix(loss=loss.item(), iou=train_iou / (loop.n + 1))

        # Epoch sonu kayıt (Checkpoint)
        if (epoch + 1) % 5 == 0:
            save_path = f"{Config.CHECKPOINT_DIR}/{Config.MODEL_NAME}_ep{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 Model kaydedildi: {save_path}")

if __name__ == "__main__":
    train()
