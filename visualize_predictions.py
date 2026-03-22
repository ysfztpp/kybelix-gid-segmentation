from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _list_images(images_dir: Path) -> list[Path]:
    return sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXT]
    )


def _overlay_mask(image: Image.Image, mask: Image.Image, alpha: float = 0.4) -> Image.Image:
    image = image.convert("RGB")
    mask = mask.convert("L")

    if mask.size != image.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)

    img_arr = np.array(image).astype(np.float32)
    mask_arr = np.array(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]

    mask_bool = mask_arr > 0
    if not np.any(mask_bool):
        return image

    color = np.array([255, 0, 0], dtype=np.float32)
    out = img_arr.copy()
    out[mask_bool] = (1.0 - alpha) * img_arr[mask_bool] + alpha * color
    return Image.fromarray(out.astype(np.uint8))


def create_overlays(
    images_dir: Union[Path, str],
    preds_dir: Union[Path, str],
    out_dir: Union[Path, str],
    n: int = 20,
    seed: int = 42,
    alpha: float = 0.4,
) -> int:
    images_dir = Path(images_dir)
    preds_dir = Path(preds_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not preds_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {preds_dir}")

    images = _list_images(images_dir)
    if not images:
        raise ValueError(f"No images found in: {images_dir}")

    pairs = []
    for img_path in images:
        mask_name = img_path.with_suffix(".png").name
        mask_path = preds_dir / mask_name
        if mask_path.is_file():
            pairs.append((img_path, mask_path))

    if not pairs:
        raise ValueError("No matching image/mask pairs found. Check filenames and folders.")

    rng = random.Random(seed)
    sample_count = min(n, len(pairs))
    sample_pairs = rng.sample(pairs, k=sample_count)

    for img_path, mask_path in sample_pairs:
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        overlay = _overlay_mask(image, mask, alpha=alpha)
        out_path = out_dir / f"{img_path.stem}_overlay.png"
        overlay.save(out_path)

    return sample_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay predicted masks on images for quick visual inspection."
    )
    parser.add_argument("--images-dir", default="test", help="Folder with input images.")
    parser.add_argument(
        "--preds-dir",
        default="test_preds",
        help="Folder with predicted masks (PNG, 0/1).",
    )
    parser.add_argument("--out-dir", default="gorsel", help="Output folder.")
    parser.add_argument("--n", type=int, default=20, help="Number of random samples to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay transparency (0-1).")
    args = parser.parse_args()

    sample_count = create_overlays(
        images_dir=args.images_dir,
        preds_dir=args.preds_dir,
        out_dir=args.out_dir,
        n=args.n,
        seed=args.seed,
        alpha=args.alpha,
    )

    print(f"Saved {sample_count} overlays to: {args.out_dir}")


if __name__ == "__main__":
    main()
