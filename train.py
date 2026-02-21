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
from src.models.model_factory import get_model
from src.utils.losses import FocalTverskyLoss
from src.utils.metrics import get_iou_score


def _configure_torch():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


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


def _build_run_name(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}"


def _resolve_resume_path(resume: str | None, model_name: str | None) -> Path | None:
    if resume is None:
        return None

    resume_path = Path(resume)
    if resume_path.is_file():
        return resume_path

    if resume_path.is_dir():
        pattern = "*_last.pth" if model_name is None else f"{model_name}_last.pth"
        candidates = sorted(resume_path.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No checkpoint files matching '{pattern}' in {resume_path}")
        return candidates[-1]

    raise FileNotFoundError(f"Resume path not found: {resume_path}")


def _resolve_resume_epoch(resume_dir: Path, resume_epoch: int, model_name: str | None) -> Path:
    if resume_epoch <= 0:
        raise ValueError("--resume-epoch must be >= 1")
    pattern = (
        f"{model_name}_epoch{resume_epoch:03d}.pth"
        if model_name
        else f"*_epoch{resume_epoch:03d}.pth"
    )
    candidates = sorted(resume_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files matching '{pattern}' in {resume_dir}")
    return candidates[-1]


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    return torch.load(checkpoint_path, map_location=device)


def _plot_metrics(history: list[dict], out_path: Path):
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_iou = [h["train_iou"] for h in history]
    val_iou = [h["val_iou"] for h in history]
    learning_rates = [h.get("learning_rate") for h in history]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 10))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(epochs, train_loss, label="train_loss")
    ax1.plot(epochs, val_loss, label="val_loss")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(epochs, train_iou, label="train_iou")
    ax2.plot(epochs, val_iou, label="val_iou")
    ax2.set_ylabel("IoU")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(epochs, learning_rates, label="learning_rate")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("LR")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _write_metrics_csv(history: list[dict], out_path: Path):
    if not history:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_iou", "val_loss", "val_iou", "learning_rate"],
        )
        writer.writeheader()
        writer.writerows(history)


def train(
    model_name=None,
    epochs=None,
    pretrained=None,
    max_train_batches=None,
    max_val_batches=None,
    resume=None,
    resume_epoch=None,
    learning_rate=None,
    reset_optimizer=False,
):
    # 1. Veri Hazırlığı
    _configure_torch()
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

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 2. Model, Loss ve Optimizer
    epochs = epochs or Config.EPOCHS
    learning_rate = learning_rate or Config.LEARNING_RATE
    pretrained = Config.PRETRAINED if pretrained is None else pretrained

    resume_path = _resolve_resume_path(resume, model_name)
    if resume_path is not None and resume_epoch is not None:
        if resume_path.is_dir():
            resume_path = _resolve_resume_epoch(resume_path, resume_epoch, model_name)
        else:
            raise ValueError("--resume-epoch can only be used when --resume points to a run directory.")
    checkpoint = None
    if resume_path is not None:
        checkpoint = _load_checkpoint(resume_path, Config.DEVICE)
        ckpt_model_name = checkpoint.get("model_name")
        if model_name is None:
            model_name = ckpt_model_name
        elif ckpt_model_name and model_name.lower() != ckpt_model_name.lower():
            raise ValueError(
                f"Checkpoint model '{ckpt_model_name}' does not match requested model '{model_name}'."
            )

    model_name = model_name or Config.MODEL_NAME
    model = get_model(model_name, n_classes=Config.NUM_CLASSES, pretrained=pretrained).to(Config.DEVICE)
    criterion = FocalTverskyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    run_name = checkpoint.get("run_name") if checkpoint else _build_run_name(model_name)
    run_dir = Path(Config.CHECKPOINT_DIR) / run_name
    result_dir = Path(Config.RESULT_DIR) / run_name
    best_val_iou = checkpoint.get("best_val_iou", -1.0) if checkpoint else -1.0
    history = checkpoint.get("history", []) if checkpoint else []
    start_epoch = checkpoint.get("epoch", 0) if checkpoint else 0

    # FP16 (Hızlı eğitim için Mixed Precision)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.DEVICE.type == "cuda")

    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if not reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"🔁 Eğitim devam ediyor: {resume_path} | Başlangıç epoch: {start_epoch + 1}")

    if start_epoch >= epochs:
        raise ValueError(f"Start epoch {start_epoch} is >= total epochs {epochs}.")

    print(f"🚀 Eğitim Başlıyor: {model_name} | Cihaz: {Config.DEVICE} | Run: {run_name}")

    for epoch in range(start_epoch, epochs):
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
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_steps, 1),
            "train_iou": train_iou / max(train_steps, 1),
            "val_loss": val_loss / max(val_steps, 1),
            "val_iou": val_iou / max(val_steps, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)

        _write_metrics_csv(history, result_dir / "metrics.csv")
        _plot_metrics(history, result_dir / "metrics.png")

        is_best = epoch_metrics["val_iou"] >= best_val_iou
        if is_best:
            best_val_iou = epoch_metrics["val_iou"]

        checkpoint = {
            "epoch": epoch + 1,
            "model_name": model_name,
            "run_name": run_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": epoch_metrics,
            "history": history,
            "best_val_iou": best_val_iou,
            "config": {
                "learning_rate": learning_rate,
                "batch_size": Config.BATCH_SIZE,
                "image_size": Config.IMAGE_SIZE,
                "pretrained": pretrained,
                "num_classes": Config.NUM_CLASSES,
            },
        }

        # Always save last checkpoint for safe resume.
        _safe_save_checkpoint(checkpoint, run_dir / f"{model_name}_last.pth")

        # Save per-epoch checkpoint.
        _safe_save_checkpoint(checkpoint, run_dir / f"{model_name}_epoch{epoch + 1:03d}.pth")

        # Save best validation IoU checkpoint.
        if is_best:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file or run directory to resume.",
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=None,
        help="When --resume is a run directory, load a specific epoch checkpoint.",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Do not load optimizer state from checkpoint (useful when changing LR).",
    )
    args = parser.parse_args()

    train(
        model_name=args.model_name,
        epochs=args.epochs,
        pretrained=args.pretrained,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        resume=args.resume,
        resume_epoch=args.resume_epoch,
        learning_rate=args.learning_rate,
        reset_optimizer=args.reset_optimizer,
    )
