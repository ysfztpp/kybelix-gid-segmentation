from __future__ import annotations

import argparse
import os
import re
import zipfile
from pathlib import Path

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
            out_path = out_dir / out_name
            _save_binary_mask(mask, str(out_path))
            saved_files.append(out_path)

    if len(saved_files) != len(test_files):
        raise RuntimeError("Output count does not match input count.")

    zip_path = str(Path(zip_name).resolve())
    _create_zip(zip_path, saved_files)
    print(f"Saved {len(saved_files)} masks to: {out_dir}")
    print(f"Zip ready: {zip_path}")


if __name__ == "__main__":
    main()
