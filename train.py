import argparse
import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from src.data.dataset import GIDDataset
from src.utils.augmentations import build_augmentations
from src.models.model_factory import get_model
from src.utils.losses import FocalTverskyLoss
from src.utils.metrics import get_iou_score

# ==========================================
# YARDIMCI FONKSİYONLAR (Optimizasyon Dahil)
# ==========================================

def _configure_torch():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

def _toggle_encoder(model, requires_grad=True):
    """Encoder katmanlarını dondurur veya çözer."""
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = requires_grad
        status = "Çözüldü" if requires_grad else "Donduruldu"
        print(f"❄️ Encoder Durumu: {status}")

def _safe_save_checkpoint(checkpoint: dict, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=save_path.parent, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, save_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

def _build_run_name(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}"

def _resolve_resume_path(resume: str | None, model_name: str | None) -> Path | None:
    if resume is None: return None
    resume_path = Path(resume)
    if resume_path.is_file(): return resume_path
    if resume_path.is_dir():
        pattern = "*_last.pth" if model_name is None else f"{model_name}_last.pth"
        candidates = sorted(resume_path.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not candidates: return None
        return candidates[-1]
    return None

def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    return torch.load(checkpoint_path, map_location=device)

def _plot_metrics(history: list[dict], out_path: Path):
    if not history: return
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.plot(epochs, [h["train_loss"] for h in history], label="train_loss")
    plt.plot(epochs, [h["val_loss"] for h in history], label="val_loss")
    plt.legend(); plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(epochs, [h["train_iou"] for h in history], label="train_iou")
    plt.plot(epochs, [h["val_iou"] for h in history], label="val_iou")
    plt.legend(); plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(epochs, [h["learning_rate"] for h in history], label="LR")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def _write_metrics_csv(history: list[dict], out_path: Path):
    if not history: return
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

# ==========================================
# ANA EĞİTİM FONKSİYONU
# ==========================================

def train(model_name=None, epochs=None, pretrained=None, max_train_batches=None, 
          max_val_batches=None, resume=None, learning_rate=None, reset_optimizer=False):
    
    _configure_torch()
    
    # 1. Veri Hazırlığı
    train_transform = build_augmentations(Config) if Config.USE_AUGMENTATION else None
    val_transform = build_augmentations(Config, is_train=False)

    train_ds = GIDDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MSK_DIR, 
                          target_color=Config.TARGET_COLOR, transform=train_transform)
    val_ds = GIDDataset(Config.VAL_IMG_DIR, Config.VAL_MSK_DIR, 
                        target_color=Config.TARGET_COLOR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    # 2. Model, Loss ve Optimizer
    model_name = model_name or Config.MODEL_NAME
    model = get_model(model_name, n_classes=Config.NUM_CLASSES, pretrained=True).to(Config.DEVICE)
    
    # Başlangıçta Encoder dondurma (Freeze)
    if Config.FREEZE_ENCODER:
        _toggle_encoder(model, requires_grad=False)

    criterion = FocalTverskyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate or Config.LEARNING_RATE)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=Config.DEVICE.type == "cuda")

    # Scheduler
    scheduler = None
    if Config.SCHEDULER == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max" if Config.EARLY_STOPPING_MONITOR == "val_iou" else "min",
            factor=Config.SCHEDULER_FACTOR, patience=Config.SCHEDULER_PATIENCE, min_lr=Config.SCHEDULER_MIN_LR
        )

    # Resume Kontrolü
    resume_path = _resolve_resume_path(resume or Config.RESUME_PATH, model_name)
    start_epoch, best_val_iou, history = 0, -1.0, []
    run_name = _build_run_name(model_name)

    if resume_path:
        checkpoint = _load_checkpoint(resume_path, Config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        if not reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint: scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_iou = checkpoint.get("best_val_iou", -1.0)
        history = checkpoint.get("history", [])
        run_name = checkpoint.get("run_name", run_name)
        print(f"🔁 Kaldığı yerden devam: {run_name} (Epoch: {start_epoch})")

    run_dir = Path(Config.CHECKPOINT_DIR) / run_name
    result_dir = Path(Config.RESULT_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 3. Eğitim Döngüsü
    print(f"🚀 Eğitim Başlıyor: {model_name} | Sanal Batch Size: {Config.BATCH_SIZE * Config.ACCUMULATION_STEPS}")

    for epoch in range(start_epoch, epochs or Config.EPOCHS):
        # Kademeli Unfreeze
        if Config.FREEZE_ENCODER and epoch == Config.UNFREEZE_EPOCH:
            _toggle_encoder(model, requires_grad=True)

        model.train()
        train_loss, train_iou, train_steps = 0.0, 0.0, 0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs or Config.EPOCHS}] (train)")
        for step, (images, masks) in enumerate(loop, start=1):
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)

            with torch.cuda.amp.autocast(enabled=Config.DEVICE.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / Config.ACCUMULATION_STEPS 

            scaler.scale(loss).backward()

            if step % Config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += (loss.item() * Config.ACCUMULATION_STEPS)
            train_iou += get_iou_score(outputs, masks)
            train_steps += 1
            loop.set_postfix(loss=train_loss/train_steps, iou=train_iou/train_steps)
            
            if max_train_batches and step >= max_train_batches: break

        # 4. Doğrulama (Validation)
        model.eval()
        val_loss, val_iou, val_steps = 0.0, 0.0, 0
        with torch.no_grad():
            vloop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs or Config.EPOCHS}] (val)")
            for step, (images, masks) in enumerate(vloop, start=1):
                images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_iou += get_iou_score(outputs, masks)
                val_steps += 1
                vloop.set_postfix(loss=val_loss/val_steps, iou=val_iou/val_steps)
                if max_val_batches and step >= max_val_batches: break

        # 5. Kayıt ve Metrikler
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss / train_steps,
            "train_iou": train_iou / train_steps,
            "val_loss": val_loss / val_steps,
            "val_iou": val_iou / val_steps,
            "learning_rate": optimizer.param_groups[0]["lr"]
        }
        history.append(epoch_metrics)
        _write_metrics_csv(history, result_dir / "metrics.csv")
        _plot_metrics(history, result_dir / "metrics.png")

        if scheduler: scheduler.step(epoch_metrics["val_iou" if Config.EARLY_STOPPING_MONITOR == "val_iou" else "val_loss"])

        # Checkpoint Kaydı
        is_best = epoch_metrics["val_iou"] > best_val_iou
        if is_best: best_val_iou = epoch_metrics["val_iou"]

        checkpoint = {
            "epoch": epoch + 1,
            "model_name": model_name,
            "run_name": run_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "best_val_iou": best_val_iou
        }
        _safe_save_checkpoint(checkpoint, run_dir / f"{model_name}_last.pth")
        if is_best: 
            _safe_save_checkpoint(checkpoint, run_dir / f"{model_name}_best.pth")
            print(f"⭐ En iyi model kaydedildi! IoU: {best_val_iou:.4f}")

    return run_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    train(model_name=args.model_name, epochs=args.epochs, resume=args.resume)
