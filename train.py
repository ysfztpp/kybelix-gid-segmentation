from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from src.data.dataset import GIDDataset
from src.models.model_factory import get_model
from src.utils.augmentations import build_augmentations
from src.utils.losses import DiceCrossEntropyBoundaryLoss
from src.utils.metrics import (
    get_binary_metrics_from_confusion,
    get_confusion_counts,
    get_iou_score,
)
from src.utils.model_outputs import get_segmentation_logits


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
        # PyTorch >=2.6 defaults to weights_only=True, which breaks full training checkpoints.
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # PyTorch <2.6 does not support weights_only argument.
        return torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise ValueError(
            "Failed to load checkpoint. Ensure the path points to a valid PyTorch "
            "checkpoint file (not an image, zip, or text file)."
        ) from exc


def _is_max_monitor(metric_name: str) -> bool:
    return metric_name != "val_loss"


def _series_from_history(history: list[dict], key: str):
    series = []
    for row in history:
        value = row.get(key)
        series.append(float("nan") if value is None else value)
    return series


def _plot_metrics(history: list[dict], out_path: Path):
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = _series_from_history(history, "train_loss")
    val_loss = _series_from_history(history, "val_loss")
    train_iou = _series_from_history(history, "train_iou")
    val_iou = _series_from_history(history, "val_iou")
    train_miou = _series_from_history(history, "train_miou")
    val_miou = _series_from_history(history, "val_miou")
    train_kappa = _series_from_history(history, "train_kappa")
    val_kappa = _series_from_history(history, "val_kappa")
    train_oa = _series_from_history(history, "train_oa")
    val_oa = _series_from_history(history, "val_oa")
    learning_rates = _series_from_history(history, "learning_rate")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 14))

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(epochs, train_loss, label="train_loss")
    ax1.plot(epochs, val_loss, label="val_loss")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(epochs, train_iou, label="train_iou")
    ax2.plot(epochs, val_iou, label="val_iou")
    ax2.plot(epochs, train_miou, label="train_miou")
    ax2.plot(epochs, val_miou, label="val_miou")
    ax2.set_ylabel("IoU / mIoU")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(epochs, train_kappa, label="train_kappa")
    ax3.plot(epochs, val_kappa, label="val_kappa")
    ax3.plot(epochs, train_oa, label="train_oa")
    ax3.plot(epochs, val_oa, label="val_oa")
    ax3.set_ylabel("Kappa / OA")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(epochs, learning_rates, label="learning_rate")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("LR")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _write_metrics_csv(history: list[dict], out_path: Path):
    if not history:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in history:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _dataset_sample_name(dataset: GIDDataset, sample_index: int) -> str:
    img_names = getattr(dataset, "img_names", None)
    if isinstance(img_names, list) and 0 <= sample_index < len(img_names):
        return str(img_names[sample_index])
    return f"sample_{sample_index:06d}"


def _safe_path_fragment(value: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value)
    return cleaned.strip("_") or "sample"


def _tensor_image_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().float()
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = image.permute(1, 2, 0)

    image_np = image.numpy()
    if image_np.ndim == 2:
        image_np = np.repeat(image_np[..., None], 3, axis=-1)
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)

    # Reverse Albumentations Normalize() defaults if tensor looks normalized.
    if image_np.min() < 0.0 or image_np.max() > 1.0:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        if image_np.shape[-1] == 3:
            image_np = image_np * std + mean

    image_np = np.clip(image_np, 0.0, 1.0)
    return (image_np * 255.0).astype(np.uint8)


def _save_binary_mask(path: Path, mask: np.ndarray):
    plt.imsave(path, (mask.astype(np.uint8) * 255), cmap="gray", vmin=0, vmax=255)


def _compute_binary_batch_stats(seg_logits: torch.Tensor, labels: torch.Tensor, threshold: float):
    if seg_logits.shape[-2:] != labels.shape[-2:]:
        seg_logits = F.interpolate(seg_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

    probs = torch.sigmoid(seg_logits)
    preds = (probs > threshold).float()
    labels = labels.float()
    if preds.shape != labels.shape:
        labels = labels.view_as(preds)

    tp = (preds * labels).sum(dim=(1, 2, 3))
    fp = (preds * (1.0 - labels)).sum(dim=(1, 2, 3))
    tn = ((1.0 - preds) * (1.0 - labels)).sum(dim=(1, 2, 3))
    fn = ((1.0 - preds) * labels).sum(dim=(1, 2, 3))

    intersection = tp
    union = preds.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)

    return preds, labels, tp, fp, tn, fn, iou


def _write_validation_samples_csv(records: list[dict], out_path: Path):
    if not records:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "image_name",
        "iou",
        "diff_ratio",
        "tp",
        "fp",
        "tn",
        "fn",
        "fg_pixels",
        "pred_pixels",
        "union_pixels",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _plot_validation_iou_hist(records: list[dict], out_path: Path):
    scores = np.array([r["iou"] for r in records], dtype=np.float32)
    if scores.size == 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bins = max(10, min(40, int(np.sqrt(scores.size)) * 2))

    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=bins, color="#2E86AB", alpha=0.85, edgecolor="black")
    plt.axvline(float(scores.mean()), color="orange", linestyle="--", linewidth=1.5, label="mean")
    plt.axvline(float(np.median(scores)), color="green", linestyle="--", linewidth=1.5, label="median")
    plt.title("Validation IoU Distribution")
    plt.xlabel("IoU")
    plt.ylabel("Sample Count")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_validation_iou_box(records: list[dict], out_path: Path):
    scores = np.array([r["iou"] for r in records], dtype=np.float32)
    if scores.size == 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.boxplot(scores, vert=False, patch_artist=True, boxprops={"facecolor": "#8FD694"})
    plt.title("Validation IoU Boxplot (Outlier Check)")
    plt.xlabel("IoU")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_validation_outlier_scatter(records: list[dict], out_path: Path):
    scores = np.array([r["iou"] for r in records], dtype=np.float32)
    diff_ratios = np.array([r["diff_ratio"] for r in records], dtype=np.float32)
    if scores.size == 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(scores, diff_ratios, alpha=0.7, s=16, color="#D1495B")
    plt.title("Validation Outliers: IoU vs Difference Ratio")
    plt.xlabel("IoU")
    plt.ylabel("Difference Ratio (FP+FN)/Pixels")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _write_validation_summary(records: list[dict], out_path: Path):
    scores = np.array([r["iou"] for r in records], dtype=np.float32)
    if scores.size == 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    q1 = float(np.percentile(scores, 25))
    q3 = float(np.percentile(scores, 75))
    iqr = q3 - q1
    outlier_threshold = q1 - 1.5 * iqr
    outlier_count = int((scores < outlier_threshold).sum())

    lines = [
        f"samples={scores.size}",
        f"mean_iou={float(scores.mean()):.6f}",
        f"std_iou={float(scores.std()):.6f}",
        f"min_iou={float(scores.min()):.6f}",
        f"p10_iou={float(np.percentile(scores, 10)):.6f}",
        f"median_iou={float(np.percentile(scores, 50)):.6f}",
        f"p90_iou={float(np.percentile(scores, 90)):.6f}",
        f"max_iou={float(scores.max()):.6f}",
        f"q1_iou={q1:.6f}",
        f"q3_iou={q3:.6f}",
        f"iqr={iqr:.6f}",
        f"outlier_threshold={outlier_threshold:.6f}",
        f"outlier_count={outlier_count}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def _build_error_overlay(
    image_rgb: np.ndarray,
    false_positive: np.ndarray,
    false_negative: np.ndarray,
) -> np.ndarray:
    overlay = image_rgb.astype(np.float32)
    highlight = np.zeros_like(overlay)
    highlight[false_positive == 1] = np.array([255, 0, 0], dtype=np.float32)
    highlight[false_negative == 1] = np.array([0, 170, 255], dtype=np.float32)

    active = (false_positive == 1) | (false_negative == 1)
    overlay[active] = 0.60 * overlay[active] + 0.40 * highlight[active]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _record_is_worse(a: dict, b: dict) -> bool:
    return (a["iou"], -a["diff_ratio"]) < (b["iou"], -b["diff_ratio"])


def _update_worst_samples(worst_samples: list[dict], candidate: dict, top_k: int) -> list[dict]:
    if top_k <= 0:
        return worst_samples
    if len(worst_samples) < top_k:
        worst_samples.append(candidate)
        return worst_samples

    best_idx = 0
    best_item = worst_samples[0]
    for idx in range(1, len(worst_samples)):
        item = worst_samples[idx]
        if _record_is_worse(best_item, item):
            best_idx = idx
            best_item = item

    if _record_is_worse(candidate, best_item):
        worst_samples[best_idx] = candidate
    return worst_samples


def _save_poor_validation_samples(sample_records: list[dict], out_dir: Path):
    if not sample_records:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    worst = sorted(sample_records, key=lambda r: (r["iou"], -r["diff_ratio"]))

    for rank, record in enumerate(worst, start=1):
        sample_idx = int(record["sample_index"])
        image_name = str(record["image_name"])
        image_rgb = record["image_rgb"]
        true_mask = record["true_mask"]
        pred_mask = record["pred_mask"]

        intersection = (pred_mask & true_mask).astype(np.uint8)
        union = (pred_mask | true_mask).astype(np.uint8)
        difference = (union & (1 - intersection)).astype(np.uint8)
        false_positive = (pred_mask & (1 - true_mask)).astype(np.uint8)
        false_negative = ((1 - pred_mask) & true_mask).astype(np.uint8)
        background = (1 - union).astype(np.uint8)

        overlay = _build_error_overlay(image_rgb, false_positive, false_negative)

        sample_stem = Path(image_name).stem or f"sample_{sample_idx:06d}"
        safe_stem = _safe_path_fragment(sample_stem)
        sample_dir = out_dir / f"rank_{rank:03d}_{safe_stem}_{sample_idx:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        plt.imsave(sample_dir / "image.png", image_rgb)
        plt.imsave(sample_dir / "error_overlay.png", overlay)
        _save_binary_mask(sample_dir / "mask_true.png", true_mask)
        _save_binary_mask(sample_dir / "mask_pred.png", pred_mask)
        _save_binary_mask(sample_dir / "mask_intersection.png", intersection)
        _save_binary_mask(sample_dir / "mask_union.png", union)
        _save_binary_mask(sample_dir / "mask_difference_union_minus_intersection.png", difference)
        _save_binary_mask(sample_dir / "mask_false_positive_extra.png", false_positive)
        _save_binary_mask(sample_dir / "mask_false_negative_missing.png", false_negative)
        _save_binary_mask(sample_dir / "mask_background.png", background)

        error_map = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
        error_map[false_positive == 1] = np.array([255, 0, 0], dtype=np.uint8)   # extra
        error_map[false_negative == 1] = np.array([0, 170, 255], dtype=np.uint8)  # missing
        plt.imsave(sample_dir / "error_map_fp_red_fn_cyan.png", error_map)

        panels = [
            (image_rgb, "Image", None),
            (overlay, "Overlay (FP red / FN cyan)", None),
            (true_mask, "GT Mask", "gray"),
            (pred_mask, "Pred Mask", "gray"),
            (error_map, "Error Map", None),
            (background, "Background", "gray"),
            (intersection, "Intersection", "gray"),
            (union, "Union", "gray"),
            (false_positive, "FP (Extra)", "gray"),
            (false_negative, "FN (Missing)", "gray"),
        ]
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for ax, (arr, title, cmap) in zip(axes.flat, panels):
            if cmap:
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(arr)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(
            f"rank={rank} | iou={record['iou']:.4f} | diff_ratio={record['diff_ratio']:.4f} | {image_name}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(sample_dir / "panel.png", dpi=150)
        plt.close(fig)

        details = [
            f"sample_index={sample_idx}",
            f"image_name={image_name}",
            f"iou={record['iou']:.6f}",
            f"diff_ratio={record['diff_ratio']:.6f}",
            f"tp={record['tp']}",
            f"fp={record['fp']}",
            f"tn={record['tn']}",
            f"fn={record['fn']}",
            f"fg_pixels={record['fg_pixels']}",
            f"pred_pixels={record['pred_pixels']}",
            f"union_pixels={record['union_pixels']}",
            "legend_fp=extra_predicted(red)",
            "legend_fn=missing_predicted(cyan)",
        ]
        (sample_dir / "metrics.txt").write_text("\n".join(details) + "\n")


def _save_validation_epoch_report(
    records: list[dict],
    worst_samples: list[dict],
    result_dir: Path,
    epoch_num: int,
    stats_dir_name: str,
):
    if not records:
        return

    epoch_dir = result_dir / stats_dir_name / f"epoch_{epoch_num:03d}"
    charts_dir = epoch_dir / "charts"
    poor_dir = epoch_dir / "poor_samples"

    _write_validation_samples_csv(records, epoch_dir / "val_sample_metrics.csv")
    _write_validation_summary(records, epoch_dir / "summary.txt")
    _plot_validation_iou_hist(records, charts_dir / "iou_histogram.png")
    _plot_validation_iou_box(records, charts_dir / "iou_boxplot.png")
    _plot_validation_outlier_scatter(records, charts_dir / "iou_vs_difference_ratio.png")
    _save_poor_validation_samples(sample_records=worst_samples, out_dir=poor_dir)


def train(
    model_name=None,
    epochs=None,
    pretrained=None,
    batch_size=None,
    num_workers=None,
    grad_accum_steps=None,
    progress_bar=None,
    log_interval=None,
    max_train_batches=None,
    max_val_batches=None,
    resume=None,
    learning_rate=None,
    reset_optimizer=False,
    metric_threshold=None,
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
    progress_bar = Config.PROGRESS_BAR if progress_bar is None else bool(progress_bar)
    log_interval = Config.LOG_INTERVAL if log_interval is None else int(log_interval)
    log_interval = max(log_interval, 1)
    metric_threshold = (
        getattr(Config, "METRIC_THRESHOLD", 0.5) if metric_threshold is None else float(metric_threshold)
    )
    reporting_enabled = bool(getattr(Config, "ENABLE_VALIDATION_REPORTING", False))
    reporting_top_k = max(int(getattr(Config, "REPORTING_TOP_K", 10)), 0)
    reporting_stats_dir = str(getattr(Config, "REPORTING_STATS_DIR_NAME", "stats"))
    use_tqdm = progress_bar and sys.stdout.isatty()

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
        mode = "max" if _is_max_monitor(monitor) else "min"
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
    early_mode = "max" if _is_max_monitor(early_monitor) else "min"
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
        f"batch={batch_size}, accum={grad_accum_steps}, effective_batch={effective_batch}, "
        f"metric_threshold={metric_threshold:.3f}"
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss, train_iou, train_steps = 0.0, 0.0, 0
        train_tp, train_fp, train_tn, train_fn = 0.0, 0.0, 0.0, 0.0
        optimizer.zero_grad(set_to_none=True)

        if use_tqdm:
            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] (train)")
        else:
            loop = train_loader
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
            train_iou += get_iou_score(outputs, masks, threshold=metric_threshold)
            tp, fp, tn, fn = get_confusion_counts(outputs, masks, threshold=metric_threshold)
            train_tp += tp
            train_fp += fp
            train_tn += tn
            train_fn += fn
            train_steps += 1

            if use_tqdm:
                loop.set_postfix(loss=loss.item(), iou=train_iou / max(train_steps, 1))
            elif step % log_interval == 0 or step == 1 or is_last_batch:
                print(
                    f"Epoch [{epoch+1}/{epochs}] (train) "
                    f"step {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f}, iou={train_iou / max(train_steps, 1):.4f}"
                )

            if max_train_batches and step >= max_train_batches:
                break

        model.eval()
        val_loss, val_iou, val_steps = 0.0, 0.0, 0
        val_tp, val_fp, val_tn, val_fn = 0.0, 0.0, 0.0, 0.0
        val_sample_records = [] if reporting_enabled else None
        worst_samples = [] if reporting_enabled and reporting_top_k > 0 else None
        seen_val_samples = 0
        with torch.no_grad():
            if use_tqdm:
                vloop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] (val)")
            else:
                vloop = val_loader
            for step, (images, masks) in enumerate(vloop, start=1):
                images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
                with torch.amp.autocast("cuda", enabled=Config.DEVICE.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                seg_logits = get_segmentation_logits(outputs)
                preds_batch, labels_batch, tp_batch, fp_batch, tn_batch, fn_batch, iou_batch = _compute_binary_batch_stats(
                    seg_logits=seg_logits,
                    labels=masks,
                    threshold=metric_threshold,
                )

                val_loss += loss.item()
                val_iou += iou_batch.mean().item()
                val_tp += tp_batch.sum().item()
                val_fp += fp_batch.sum().item()
                val_tn += tn_batch.sum().item()
                val_fn += fn_batch.sum().item()

                if reporting_enabled and val_sample_records is not None:
                    batch_size_curr = int(iou_batch.shape[0])
                    for batch_idx in range(batch_size_curr):
                        sample_index = seen_val_samples + batch_idx
                        tp_val = float(tp_batch[batch_idx].item())
                        fp_val = float(fp_batch[batch_idx].item())
                        tn_val = float(tn_batch[batch_idx].item())
                        fn_val = float(fn_batch[batch_idx].item())
                        total_pixels = tp_val + fp_val + tn_val + fn_val
                        diff_ratio = (fp_val + fn_val) / max(total_pixels, 1e-7)
                        fg_pixels = tp_val + fn_val
                        pred_pixels = tp_val + fp_val
                        union_pixels = tp_val + fp_val + fn_val

                        val_sample_records.append(
                            {
                                "sample_index": sample_index,
                                "image_name": _dataset_sample_name(val_ds, sample_index),
                                "iou": float(iou_batch[batch_idx].item()),
                                "diff_ratio": float(diff_ratio),
                                "tp": tp_val,
                                "fp": fp_val,
                                "tn": tn_val,
                                "fn": fn_val,
                                "fg_pixels": fg_pixels,
                                "pred_pixels": pred_pixels,
                                "union_pixels": union_pixels,
                            }
                        )

                        if worst_samples is not None:
                            image_rgb = _tensor_image_to_uint8(images[batch_idx])
                            true_mask = (labels_batch[batch_idx, 0] > 0.5).to(torch.uint8).detach().cpu().numpy()
                            pred_mask = (preds_batch[batch_idx, 0] > 0.5).to(torch.uint8).detach().cpu().numpy()
                            worst_samples = _update_worst_samples(
                                worst_samples=worst_samples,
                                candidate={
                                    "sample_index": sample_index,
                                    "image_name": _dataset_sample_name(val_ds, sample_index),
                                    "iou": float(iou_batch[batch_idx].item()),
                                    "diff_ratio": float(diff_ratio),
                                    "tp": tp_val,
                                    "fp": fp_val,
                                    "tn": tn_val,
                                    "fn": fn_val,
                                    "fg_pixels": fg_pixels,
                                    "pred_pixels": pred_pixels,
                                    "union_pixels": union_pixels,
                                    "image_rgb": image_rgb,
                                    "true_mask": true_mask,
                                    "pred_mask": pred_mask,
                                },
                                top_k=reporting_top_k,
                            )
                    seen_val_samples += batch_size_curr

                val_steps += 1
                is_last_val_batch = (step == len(val_loader)) or (max_val_batches and step >= max_val_batches)
                if use_tqdm:
                    vloop.set_postfix(loss=loss.item(), iou=val_iou / max(val_steps, 1))
                elif step % log_interval == 0 or step == 1 or is_last_val_batch:
                    print(
                        f"Epoch [{epoch+1}/{epochs}] (val) "
                        f"step {step}/{len(val_loader)} "
                        f"loss={loss.item():.4f}, iou={val_iou / max(val_steps, 1):.4f}"
                    )

                if max_val_batches and step >= max_val_batches:
                    break

        train_bin_metrics = get_binary_metrics_from_confusion(train_tp, train_fp, train_tn, train_fn)
        val_bin_metrics = get_binary_metrics_from_confusion(val_tp, val_fp, val_tn, val_fn)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_steps, 1),
            "train_iou": train_iou / max(train_steps, 1),
            "train_miou": train_bin_metrics["miou"],
            "train_oa": train_bin_metrics["oa"],
            "train_kappa": train_bin_metrics["kappa"],
            "train_iou_fg": train_bin_metrics["iou_fg"],
            "train_iou_bg": train_bin_metrics["iou_bg"],
            "val_loss": val_loss / max(val_steps, 1),
            "val_iou": val_iou / max(val_steps, 1),
            "val_miou": val_bin_metrics["miou"],
            "val_oa": val_bin_metrics["oa"],
            "val_kappa": val_bin_metrics["kappa"],
            "val_iou_fg": val_bin_metrics["iou_fg"],
            "val_iou_bg": val_bin_metrics["iou_bg"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "metric_threshold": metric_threshold,
        }
        history.append(epoch_metrics)

        _write_metrics_csv(history, result_dir / "metrics.csv")
        _plot_metrics(history, result_dir / "metrics.png")
        if reporting_enabled and val_sample_records:
            _save_validation_epoch_report(
                records=val_sample_records,
                worst_samples=worst_samples or [],
                result_dir=result_dir,
                epoch_num=epoch + 1,
                stats_dir_name=reporting_stats_dir,
            )

        is_best = epoch_metrics["val_iou"] >= best_val_iou
        if is_best:
            best_val_iou = epoch_metrics["val_iou"]

        if scheduler:
            scheduler.step(epoch_metrics[early_monitor])

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
                "progress_bar": progress_bar,
                "log_interval": log_interval,
                "metric_threshold": metric_threshold,
                "image_size": Config.IMAGE_SIZE,
                "pretrained": pretrained,
                "num_classes": Config.NUM_CLASSES,
                "edge_loss_weight": Config.EDGE_LOSS_WEIGHT,
                "edge_target_method": Config.EDGE_TARGET_METHOD,
                "edge_sobel_threshold": Config.EDGE_SOBEL_THRESHOLD,
                "enable_validation_reporting": reporting_enabled,
                "reporting_top_k": reporting_top_k,
                "reporting_stats_dir_name": reporting_stats_dir,
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
            f"train_miou={epoch_metrics['train_miou']:.4f}, train_kappa={epoch_metrics['train_kappa']:.4f}, "
            f"train_oa={epoch_metrics['train_oa']:.4f}, val_loss={epoch_metrics['val_loss']:.4f}, "
            f"val_iou={epoch_metrics['val_iou']:.4f}, val_miou={epoch_metrics['val_miou']:.4f}, "
            f"val_kappa={epoch_metrics['val_kappa']:.4f}, val_oa={epoch_metrics['val_oa']:.4f}"
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
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--no-progress-bar", dest="progress_bar", action="store_false")
    parser.set_defaults(progress_bar=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--metric-threshold", type=float, default=None)
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
        progress_bar=args.progress_bar,
        log_interval=args.log_interval,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        resume=args.resume,
        learning_rate=args.learning_rate,
        reset_optimizer=args.reset_optimizer,
        metric_threshold=args.metric_threshold,
    )
