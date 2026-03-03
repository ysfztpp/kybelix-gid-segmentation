import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from src.data.dataset import GIDDataset
from src.models.model_factory import get_model
from src.utils.augmentations import build_augmentations
from src.utils.metrics import get_binary_metrics_from_confusion, get_confusion_counts_from_probs
from src.utils.model_outputs import get_segmentation_logits


def _build_thresholds(min_threshold: float, max_threshold: float, step: float):
    if step <= 0:
        raise ValueError("Threshold step must be > 0.")
    if min_threshold > max_threshold:
        raise ValueError("min-threshold cannot be greater than max-threshold.")
    thresholds = np.arange(min_threshold, max_threshold + (0.5 * step), step)
    return [float(f"{t:.6f}") for t in thresholds]


def _write_csv(rows: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "kappa", "miou", "oa", "iou_fg", "iou_bg"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_rows(rows: list[dict], best_threshold: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [r["threshold"] for r in rows]
    kappa = [r["kappa"] for r in rows]
    miou = [r["miou"] for r in rows]
    oa = [r["oa"] for r in rows]
    iou_fg = [r["iou_fg"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, kappa, label="kappa", linewidth=2)
    plt.plot(thresholds, miou, label="miou", linewidth=2)
    plt.plot(thresholds, oa, label="oa", linewidth=2)
    plt.plot(thresholds, iou_fg, label="iou_fg", linewidth=2)
    plt.axvline(best_threshold, linestyle="--", linewidth=1.5, label=f"best={best_threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Find best segmentation threshold by validation Kappa.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth/.pt/.ckpt file.")
    parser.add_argument("--model-name", type=str, default=None, help="Optional model override.")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=Config.NUM_WORKERS)
    parser.add_argument("--min-threshold", type=float, default=0.10)
    parser.add_argument("--max-threshold", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results/threshold_sweeps")
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    thresholds = _build_thresholds(args.min_threshold, args.max_threshold, args.step)
    device = Config.DEVICE

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = args.model_name or checkpoint.get("model_name") or Config.MODEL_NAME

    model = get_model(model_name, n_classes=Config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_transform = build_augmentations(Config, is_train=False) if Config.USE_AUGMENTATION else None
    val_ds = GIDDataset(
        Config.VAL_IMG_DIR,
        Config.VAL_MSK_DIR,
        target_color=Config.TARGET_COLOR,
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=Config.PIN_MEMORY,
    )

    counts = {threshold: [0.0, 0.0, 0.0, 0.0] for threshold in thresholds}
    use_tqdm = args.progress_bar and bool(getattr(sys.stdout, "isatty", lambda: False)())

    with torch.no_grad():
        iterator = tqdm(val_loader, desc="Threshold sweep") if use_tqdm else val_loader
        for step, (images, masks) in enumerate(iterator, start=1):
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)
            logits = get_segmentation_logits(outputs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits)

            for threshold in thresholds:
                tp, fp, tn, fn = get_confusion_counts_from_probs(probs, masks, threshold=threshold)
                bucket = counts[threshold]
                bucket[0] += tp
                bucket[1] += fp
                bucket[2] += tn
                bucket[3] += fn

            if args.max_val_batches and step >= args.max_val_batches:
                break

    rows = []
    for threshold in thresholds:
        tp, fp, tn, fn = counts[threshold]
        metrics = get_binary_metrics_from_confusion(tp, fp, tn, fn)
        rows.append(
            {
                "threshold": threshold,
                "kappa": metrics["kappa"],
                "miou": metrics["miou"],
                "oa": metrics["oa"],
                "iou_fg": metrics["iou_fg"],
                "iou_bg": metrics["iou_bg"],
            }
        )

    best = max(rows, key=lambda r: (r["kappa"], r["miou"], r["oa"], r["iou_fg"]))
    run_name = checkpoint.get("run_name") or checkpoint_path.stem
    out_dir = Path(args.output_dir) / run_name
    csv_path = out_dir / "threshold_metrics.csv"
    png_path = out_dir / "threshold_metrics.png"

    _write_csv(rows, csv_path)
    _plot_rows(rows, best["threshold"], png_path)

    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")
    print(
        "Best threshold by Kappa: "
        f"{best['threshold']:.3f} | kappa={best['kappa']:.4f}, miou={best['miou']:.4f}, oa={best['oa']:.4f}"
    )


if __name__ == "__main__":
    main()
