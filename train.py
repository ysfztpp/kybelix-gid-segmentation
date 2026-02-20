import os
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from src.data.dataset import GIDDataset
from src.models.model_factory import get_model
from src.utils.losses import FocalTverskyLoss
from src.utils.metrics import get_iou_score


def _safe_save_checkpoint(checkpoint: dict, save_path: Path):
    """Write checkpoint atomically to avoid partial/corrupted files."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=save_path.parent, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, save_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _build_run_name(model_name: str, learning_rate: float) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_lr{learning_rate:g}_{timestamp}"


def train(
    model_name=None,
    epochs=None,
    pretrained=None,
    max_train_batches=None,
    max_val_batches=None,
):
    # 1. Veri Hazırlığı
    train_ds = GIDDataset(
        Config.TRAIN_IMG_DIR,
        Config.TRAIN_MSK_DIR,
        target_color=Config.TARGET_COLOR,
    )
    val_ds = GIDDataset(
        Config.VAL_IMG_DIR,
        Config.VAL_MSK_DIR,
        target_color=Config.TARGET_COLOR,
    )

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 2. Model, Loss ve Optimizer
    model_name = model_name or Config.MODEL_NAME
    epochs = epochs or Config.EPOCHS
    pretrained = Config.PRETRAINED if pretrained is None else pretrained

    model = get_model(model_name, pretrained=pretrained).to(Config.DEVICE)
    criterion = FocalTverskyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    run_name = _build_run_name(model_name, Config.LEARNING_RATE)
    run_dir = Path(Config.CHECKPOINT_DIR) / run_name
    best_val_iou = -1.0

    # FP16 (Hızlı eğitim için Mixed Precision)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.DEVICE.type == "cuda")

    print(f"🚀 Eğitim Başlıyor: {model_name} | Cihaz: {Config.DEVICE} | Run: {run_name}")

    for epoch in range(epochs):
        model.train()
        train_loss, train_iou, train_steps = 0.0, 0.0, 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] (train)")
        for step, (images, masks) in enumerate(loop, start=1):
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)

            # Forward pass with Mixed Precision
            with torch.cuda.amp.autocast(enabled=Config.DEVICE.type == "cuda"):
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
            train_steps += 1

            loop.set_postfix(loss=loss.item(), iou=train_iou / max(train_steps, 1))

            if max_train_batches and step >= max_train_batches:
                break

        model.eval()
        val_loss, val_iou, val_steps = 0.0, 0.0, 0
        with torch.no_grad():
            vloop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] (val)")
            for step, (images, masks) in enumerate(vloop, start=1):
                images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_iou += get_iou_score(outputs, masks)
                val_steps += 1
                vloop.set_postfix(loss=loss.item(), iou=val_iou / max(val_steps, 1))

                if max_val_batches and step >= max_val_batches:
                    break

        epoch_metrics = {
            "train_loss": train_loss / max(train_steps, 1),
            "train_iou": train_iou / max(train_steps, 1),
            "val_loss": val_loss / max(val_steps, 1),
            "val_iou": val_iou / max(val_steps, 1),
        }

        checkpoint = {
            "epoch": epoch + 1,
            "model_name": model_name,
            "run_name": run_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": epoch_metrics,
            "config": {
                "learning_rate": Config.LEARNING_RATE,
                "batch_size": Config.BATCH_SIZE,
                "image_size": Config.IMAGE_SIZE,
                "pretrained": pretrained,
            },
        }

        # Always save last checkpoint for safe resume.
        _safe_save_checkpoint(checkpoint, run_dir / f"{model_name}_last.pth")

        # Save per-epoch checkpoint.
        _safe_save_checkpoint(checkpoint, run_dir / f"{model_name}_epoch{epoch + 1:03d}.pth")

        # Save best validation IoU checkpoint.
        if epoch_metrics["val_iou"] >= best_val_iou:
            best_val_iou = epoch_metrics["val_iou"]
            best_ckpt = run_dir / f"{model_name}_best.pth"
            _safe_save_checkpoint(checkpoint, best_ckpt)
            print(f"⭐ Yeni en iyi model kaydedildi: {best_ckpt} (IoU: {best_val_iou:.4f})")

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={epoch_metrics['train_loss']:.4f}, train_iou={epoch_metrics['train_iou']:.4f}, "
            f"val_loss={epoch_metrics['val_loss']:.4f}, val_iou={epoch_metrics['val_iou']:.4f}"
        )

    return run_name


if __name__ == "__main__":
    train()
