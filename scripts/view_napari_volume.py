#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vesuvius.visualization.napari_viewer import view_volume_with_mask

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Open a Vesuvius CT volume (+ optional mask) in napari."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("~/vesuvius-scroll-detection/data/raw/vesuvius").expanduser(),
        help="Root folder containing train_images/ and train_labels/ (default: repo data path).",
    )
    p.add_argument(
        "--sid",
        type=str,
        required=True,
        help="Sample id (e.g. 8862040).",
    )
    p.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Which folders to use. train -> train_images/train_labels, test -> test_images/(optional test_labels).",
    )
    p.add_argument(
        "--ndisplay",
        type=int,
        default=3,
        choices=[2, 3],
        help="napari display mode (2 or 3). Default 3.",
    )
    p.add_argument(
        "--mask-mode",
        choices=["surface", "volume"],
        default="surface",
        help="How to render the mask if present. surface (mesh) is best for thin sheets.",
    )
    p.add_argument(
        "--axis",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="If viewing a single 2D slice later, which axis to slice along (not required for full 3D view).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root
    sid: str = args.sid

    if args.split == "train":
        img_path = root / "train_images" / f"{sid}.tif"
        mask_path = root / "train_labels" / f"{sid}.tif"
    else:
        img_path = root / "test_images" / f"{sid}.tif"
        mask_path = root / "test_labels" / f"{sid}.tif"  # may not exist

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    mask = mask_path if mask_path.exists() else None

    view_volume_with_mask(
        img_path,
        mask,
        ndisplay=args.ndisplay,
        mask_mode=args.mask_mode,
    )


if __name__ == "__main__":
    main()