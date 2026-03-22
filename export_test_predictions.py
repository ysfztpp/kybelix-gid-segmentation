from __future__ import annotations

import argparse
import os
import re
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from src.models.model_factory import get_model
from src.utils.model_outputs import get_segmentation_logits


def _load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _prepare_image(path: str, size: int) -> torch.Tensor:
    img_bytes = np.fromfile(path, np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _save_binary_mask(mask: np.ndarray, out_path: str) -> None:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError("Mask must be single-channel.")
    cv2.imwrite(out_path, mask)


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read mask: {path}")
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    return mask


def _validate_mask(path: Path, size: int) -> list[str]:
    issues: list[str] = []
    try:
        mask = _read_mask(path)
    except ValueError as exc:
        return [str(exc)]

    if mask.ndim != 2:
        issues.append(f"Mask must be single-channel, got shape {mask.shape}.")

    if mask.shape != (size, size):
        issues.append(f"Mask size must be {size}x{size}, got {mask.shape[0]}x{mask.shape[1]}.")

    if not np.issubdtype(mask.dtype, np.integer):
        issues.append(f"Mask dtype must be integer, got {mask.dtype}.")

    bad = (mask != 0) & (mask != 1)
    if np.any(bad):
        bad_vals = np.unique(mask[bad])
        issues.append(f"Mask has invalid values: {bad_vals.tolist()}")

    return issues


def _validate_predictions(
    test_files: list[Path],
    out_dir: Path,
    size: int,
    report_path: Optional[Path],
    verbose: bool,
) -> None:
    expected_names = [p.with_suffix(".png").name for p in test_files]
    expected_set = set(expected_names)

    lines: list[str] = []
    ok_count = 0
    fail_count = 0
    missing_count = 0

    for name in expected_names:
        mask_path = out_dir / name
        if not mask_path.is_file():
            missing_count += 1
            msg = f"FAIL {name}: missing"
            lines.append(msg)
            if verbose:
                print(msg)
            continue

        issues = _validate_mask(mask_path, size)
        if issues:
            fail_count += 1
            msg = f"FAIL {name}: " + "; ".join(issues)
            lines.append(msg)
            if verbose:
                print(msg)
        else:
            ok_count += 1
            msg = f"OK   {name}"
            lines.append(msg)
            if verbose:
                print(msg)

    extra_files = sorted([p.name for p in out_dir.iterdir() if p.is_file() and p.name not in expected_set])
    if extra_files:
        lines.append(f"WARN extra files in output dir (not in test set): {extra_files}")

    summary = (
        f"Summary: total={len(expected_names)} ok={ok_count} fail={fail_count} "
        f"missing={missing_count} extra={len(extra_files)}"
    )
    lines.append(summary)
    print(summary)

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")

    if fail_count > 0 or missing_count > 0:
        raise RuntimeError(
            "Validation failed. See report for details."
            + (f" Report: {report_path}" if report_path else "")
        )


def _validate_zip_name(zip_name: str) -> str:
    if not zip_name.lower().endswith(".zip"):
        zip_name = f"{zip_name}.zip"
    if " " in zip_name:
        raise ValueError("Zip filename must not contain spaces.")
    if not re.fullmatch(r"[A-Za-z0-9@._+\-]+\.zip", zip_name):
        raise ValueError(
            "Zip filename contains invalid characters. "
            "Use only letters, numbers, @ . _ + -"
        )
    return zip_name


def _collect_test_images(test_dir: str) -> list[Path]:
    allowed_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    test_path = Path(test_dir)
    files = [p for p in sorted(test_path.iterdir()) if p.is_file() and p.suffix.lower() in allowed_ext]
    if not files:
        raise ValueError(f"No test images found in: {test_dir}")
    return files


def _create_zip(zip_path: str, file_paths: list[Path]) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in file_paths:
            zf.write(file_path, arcname=file_path.name)


def main():
    parser = argparse.ArgumentParser(description="Export test predictions as 0/1 PNG masks (512x512).")
    parser.add_argument("--ckpt", required=True, help="Path to the final checkpoint (.pth).")
    parser.add_argument(
        "--test-dir",
        default=os.path.join(Config.BASE_PATH, "unmasked", "test"),
        help="Directory containing test images.",
    )
    parser.add_argument("--out-dir", default="test_preds", help="Output directory for PNG masks.")
    parser.add_argument(
        "--zip-name",
        required=True,
        help="Zip filename: Team+Leader+Email+Phone.zip (no spaces).",
    )
    parser.add_argument("--size", type=int, default=512, help="Output size (pixels). Must be 512.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(getattr(Config, "METRIC_THRESHOLD", 0.5)),
        help="Threshold for binary mask (sigmoid).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of saved PNG masks.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to write validation report (txt). Default: <out-dir>/validation_report.txt",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print validation status for every file (line-by-line).",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip overlay visualization generation.",
    )
    parser.add_argument("--viz-dir", default="gorsel", help="Output folder for overlays.")
    parser.add_argument("--viz-samples", type=int, default=20, help="Number of overlays to generate.")
    parser.add_argument("--viz-seed", type=int, default=42, help="Random seed for overlays.")
    parser.add_argument("--viz-alpha", type=float, default=0.4, help="Overlay transparency (0-1).")
    args = parser.parse_args()

    if args.size != 512:
        raise ValueError("Output size must be 512x512 to match submission requirements.")

    zip_name = _validate_zip_name(args.zip_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = Config.DEVICE
    checkpoint = _load_checkpoint(args.ckpt, device)
    model_name = checkpoint.get("model_name", Config.MODEL_NAME)

    model = get_model(model_name, n_classes=Config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_files = _collect_test_images(args.test_dir)
    saved_files: list[Path] = []
    seen_names: set[str] = set()

    with torch.no_grad():
        for img_path in test_files:
            x = _prepare_image(str(img_path), args.size).to(device)
            outputs = model(x)
            seg_logits = get_segmentation_logits(outputs)

            if seg_logits.shape[-2:] != (args.size, args.size):
                seg_logits = F.interpolate(seg_logits, size=(args.size, args.size), mode="bilinear", align_corners=False)

            if seg_logits.shape[1] == 1:
                probs = torch.sigmoid(seg_logits)
                mask = (probs >= args.threshold).squeeze().cpu().numpy().astype(np.uint8)
            else:
                probs = torch.softmax(seg_logits, dim=1)
                preds = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                mask = (preds == 1).astype(np.uint8)

            out_name = img_path.with_suffix(".png").name
            if out_name in seen_names:
                raise ValueError(
                    f"Duplicate output filename detected: {out_name}. "
                    "Check test set for duplicate stems."
                )
            seen_names.add(out_name)
            out_path = out_dir / out_name
            _save_binary_mask(mask, str(out_path))
            saved_files.append(out_path)

    if len(saved_files) != len(test_files):
        raise RuntimeError("Output count does not match input count.")

    if not args.no_validate:
        report_path = Path(args.report) if args.report else out_dir / "validation_report.txt"
        _validate_predictions(test_files, out_dir, args.size, report_path, args.verbose)

    zip_path = str(Path(zip_name).resolve())
    _create_zip(zip_path, saved_files)
    print(f"Saved {len(saved_files)} masks to: {out_dir}")
    print(f"Zip ready: {zip_path}")

    if not args.no_visualize:
        try:
            from visualize_predictions import create_overlays
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("Failed to import visualization helper.") from exc

        sample_count = create_overlays(
            images_dir=args.test_dir,
            preds_dir=out_dir,
            out_dir=args.viz_dir,
            n=args.viz_samples,
            seed=args.viz_seed,
            alpha=args.viz_alpha,
        )
        print(f"Saved {sample_count} overlays to: {args.viz_dir}")


if __name__ == "__main__":
    main()
