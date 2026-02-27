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
from src.utils.losses import DiceCrossEntropyBoundaryLoss
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


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if checkpoint_path.suffix.lower() not in {".pth", ".pt", ".ckpt"}:
        raise ValueError(
            "Resume path must be a PyTorch checkpoint file (.pth/.pt/.ckpt). "
            f"Got: {checkpoint_path}"
        )

    try:
        return torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise ValueError(
            "Failed to load checkpoint. Ensure the path points to a valid PyTorch "
            "checkpoint file (not an image, zip, or text file)."
        ) from exc


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
    batch_size=None,
    num_workers=None,
    grad_accum_steps=None,
    max_train_batches=None,
    max_val_batches=None,
    resume=None,
    learning_rate=None,
    reset_optimizer=False,
):
    # 1. Veri Hazırlığı
    _configure_torch()
    train_transform = build_augmentations(Config) if Config.USE_AUGMENTATION else None
    val_transform = build_augmentations(Config, is_train=False) if Config.USE_AUGMENTATION else None

    train_ds = GIDDataset(
        Config.TRAIN_IMG_DIR,
        Config.TRAIN_MSK_DIR,
        target_color=Config.TARGET_COLOR,
        transform=train_transform,
    )
    val_ds = GIDDataset(
        Config.VAL_IMG_DIR,
        Config.VAL_MSK_DIR,
        target_color=Config.TARGET_COLOR,
        transform=val_transform,
    )

    batch_size = batch_size or Config.BATCH_SIZE
    num_workers = Config.NUM_WORKERS if num_workers is None else num_workers
    grad_accum_steps = grad_accum_steps or Config.GRAD_ACCUM_STEPS
    grad_accum_steps = max(int(grad_accum_steps), 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
    )

    # 2. Model, Loss ve Optimizer
    epochs = epochs or Config.EPOCHS
    learning_rate = learning_rate or Config.LEARNING_RATE
    pretrained = Config.PRETRAINED if pretrained is None else pretrained

    resume = Config.RESUME_PATH if resume is None else resume
    reset_optimizer = reset_optimizer or Config.RESET_OPTIMIZER

    resume_path = _resolve_resume_path(resume, model_name)
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
    criterion = DiceCrossEntropyBoundaryLoss(
        lambda_edge=Config.EDGE_LOSS_WEIGHT,
        edge_method=Config.EDGE_TARGET_METHOD,
        sobel_threshold=Config.EDGE_SOBEL_THRESHOLD,
    )
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = None
    if Config.SCHEDULER == "plateau":
        monitor = Config.EARLY_STOPPING_MONITOR
        mode = "max" if monitor == "val_iou" else "min"
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.SCHEDULER_MIN_LR,
        )

    run_name = checkpoint.get("run_name") if checkpoint else _build_run_name(model_name)
    run_dir = Path(Config.CHECKPOINT_DIR) / run_name
    local_run_dir = Path(Config.LOCAL_CHECKPOINT_DIR) / run_name
    if not Config.SAVE_RUNS_TO_DRIVE:
        local_run_dir = run_dir
    result_dir = Path(Config.RESULT_DIR) / run_name
    best_val_iou = checkpoint.get("best_val_iou", -1.0) if checkpoint else -1.0
    history = checkpoint.get("history", []) if checkpoint else []
    start_epoch = checkpoint.get("epoch", 0) if checkpoint else 0
    early_state = checkpoint.get("early_stopping") if checkpoint else None
    early_monitor = Config.EARLY_STOPPING_MONITOR
    early_mode = "max" if early_monitor == "val_iou" else "min"
    if early_state:
        early_best = early_state.get("best")
        early_bad_epochs = early_state.get("bad_epochs", 0)
    else:
        early_best = -float("inf") if early_mode == "max" else float("inf")
        early_bad_epochs = 0

    # FP16 (Hızlı eğitim için Mixed Precision)
    scaler = torch.amp.GradScaler("cuda", enabled=Config.DEVICE.type == "cuda")

    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if not reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"🔁 Eğitim devam ediyor: {resume_path} | Başlangıç epoch: {start_epoch + 1}")

    if start_epoch >= epochs:
        raise ValueError(f"Start epoch {start_epoch} is >= total epochs {epochs}.")

    effective_batch = batch_size * grad_accum_steps
    print(
        f"🚀 Eğitim Başlıyor: {model_name} | Cihaz: {Config.DEVICE} | Run: {run_name} | "
        f"batch={batch_size}, accum={grad_accum_steps}, effective_batch={effective_batch}"
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss, train_iou, train_steps = 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] (train)")
        for step, (images, masks) in enumerate(loop, start=1):
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)

            # Forward pass with Mixed Precision
            with torch.amp.autocast("cuda", enabled=Config.DEVICE.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)

            # Backward pass (supports gradient accumulation for large effective batch size)
            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

            is_last_batch = (step == len(train_loader)) or (max_train_batches and step >= max_train_batches)
            if step % grad_accum_steps == 0 or is_last_batch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

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
                with torch.amp.autocast("cuda", enabled=Config.DEVICE.type == "cuda"):
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

        if scheduler:
            if early_monitor == "val_iou":
                scheduler.step(epoch_metrics["val_iou"])
            else:
                scheduler.step(epoch_metrics["val_loss"])

        if Config.EARLY_STOPPING:
            current = epoch_metrics[early_monitor]
            improved = (
                current >= early_best + Config.EARLY_STOPPING_MIN_DELTA
                if early_mode == "max"
                else current <= early_best - Config.EARLY_STOPPING_MIN_DELTA
            )
            if improved:
                early_best = current
                early_bad_epochs = 0
            else:
                early_bad_epochs += 1

        checkpoint = {
            "epoch": epoch + 1,
            "model_name": model_name,
            "run_name": run_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": epoch_metrics,
            "history": history,
            "best_val_iou": best_val_iou,
            "early_stopping": {
                "monitor": early_monitor,
                "mode": early_mode,
                "best": early_best,
                "bad_epochs": early_bad_epochs,
            },
            "config": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": effective_batch,
                "num_workers": num_workers,
                "image_size": Config.IMAGE_SIZE,
                "pretrained": pretrained,
                "num_classes": Config.NUM_CLASSES,
                "edge_loss_weight": Config.EDGE_LOSS_WEIGHT,
                "edge_target_method": Config.EDGE_TARGET_METHOD,
                "edge_sobel_threshold": Config.EDGE_SOBEL_THRESHOLD,
            },
        }

        # Always save last checkpoint for safe resume (Drive if enabled).
        _safe_save_checkpoint(checkpoint, run_dir / f"{model_name}_last.pth")

        # Save per-epoch checkpoint locally.
        if Config.SAVE_EPOCHS_LOCALLY:
            _safe_save_checkpoint(
                checkpoint, local_run_dir / f"{model_name}_epoch{epoch + 1:03d}.pth"
            )

        # Save best validation IoU checkpoint (Drive if enabled).
        if is_best:
            best_ckpt = run_dir / f"{model_name}_best.pth"
            _safe_save_checkpoint(checkpoint, best_ckpt)
            print(f"⭐ Yeni en iyi model kaydedildi: {best_ckpt} (IoU: {best_val_iou:.4f})")

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={epoch_metrics['train_loss']:.4f}, train_iou={epoch_metrics['train_iou']:.4f}, "
            f"val_loss={epoch_metrics['val_loss']:.4f}, val_iou={epoch_metrics['val_iou']:.4f}"
        )

        if Config.EARLY_STOPPING and early_bad_epochs >= Config.EARLY_STOPPING_PATIENCE:
            print(
                f"🛑 Early stopping: no improvement in {early_monitor} for "
                f"{Config.EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    return run_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
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
        "--reset-optimizer",
        action="store_true",
        help="Do not load optimizer state from checkpoint (useful when changing LR).",
    )
    args = parser.parse_args()

    train(
        model_name=args.model_name,
        epochs=args.epochs,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grad_accum_steps=args.grad_accum_steps,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        resume=args.resume,
        learning_rate=args.learning_rate,
        reset_optimizer=args.reset_optimizer,
    )
