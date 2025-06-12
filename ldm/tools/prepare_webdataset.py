#!/usr/bin/env python3
"""prepare_webdataset.py

Convert PNG/JSON pairs into **WebDataset** shards.  Each output record
contains:

* ``<key>.jpg`` - RGB JPEG (quality set by ``--jpeg_quality``)
* ``<key>.json`` - consolidated metadata (url, caption, sha256, dimensions)

Highlights
~~~~~~~~~~
* Strips oversized XMP/XML chunks so Pillow never raises
  ``ValueError: XMP data is too long``.
* JPEG quality is now configurable via **command-line** (``--jpeg_quality``),
  defaulting to **95**.

Usage example
-------------
```bash
python prepare_webdataset.py \
  --input_dirs 00000 00001 \
  --output_dir shards \
  --image_size 256 \
  --jpeg_quality 90
```

Dependencies
------------
* Pillow      → `pip install Pillow`
* webdataset  → `pip install webdataset`
* loguru      → `pip install loguru`
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image
from loguru import logger
import webdataset as wds

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_data_paths(input_dirs: Iterable[str]) -> List[Tuple[Path, Path]]:
    """Return list of (png, json) path tuples discovered under *input_dirs*."""
    pairs: List[Tuple[Path, Path]] = []
    for directory in input_dirs:
        for png in sorted(Path(directory).rglob("*.png")):
            meta_path = png.with_suffix(".json")
            if meta_path.exists():
                pairs.append((png, meta_path))
            else:
                logger.warning("Missing JSON for {} - skipping", png)
    logger.info("Indexed {} PNG/JSON pairs from {} folders", len(pairs), len(input_dirs))
    return pairs


def resize_image(img: Image.Image, image_size: int) -> Image.Image:
    """Resize *img* so its shortest side equals *image_size* using Lanczos."""
    w, h = img.size
    scale = image_size / float(min(w, h))
    new_size = (round(w * scale), round(h * scale))
    return img.resize(new_size, resample=Image.LANCZOS)


def strip_problematic_metadata(pil_img: Image.Image) -> None:
    """Remove XMP/XML blocks that exceed JPEG limits."""
    for key in ("xmp", "xml"):
        if key in pil_img.info:
            pil_img.info.pop(key, None)


def process_sample(
    png_path: Path,
    json_path: Path,
    image_size: int,
    jpeg_quality: int,
) -> dict | None:
    """Convert one PNG/JSON pair into a WebDataset sample dict."""
    try:
        # --- load metadata --------------------------------------------------
        with json_path.open("r", encoding="utf-8") as f:
            meta_in = json.load(f)

        # --- process image --------------------------------------------------
        with Image.open(png_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            img = resize_image(img, image_size)
            resized_w, resized_h = img.size

            strip_problematic_metadata(img)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            jpg_bytes = buf.getvalue()

        # --- merge + encode metadata ---------------------------------------
        meta_out = {
            "url": meta_in.get("url", ""),
            "caption": meta_in.get("caption", ""),
            "sha256": meta_in.get("sha256", ""),
            "width": resized_w,
            "height": resized_h,
            "original_width": orig_w,
            "original_height": orig_h,
        }
        json_bytes = json.dumps(meta_out, ensure_ascii=False).encode("utf-8")

        key = str(meta_in.get("key", png_path.stem))
        return {"__key__": key, "jpg": jpg_bytes, "json": json_bytes}

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to process {}: {}", png_path, exc)
        return None


def write_shards(
    pairs: List[Tuple[Path, Path]],
    output_dir: str,
    *,
    maxcount: int,
    maxsize: int,
    image_size: int,
    jpeg_quality: int,
) -> None:
    """Stream *pairs* into shard files under *output_dir*."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%05d.tar")

    with wds.ShardWriter(pattern, maxcount=maxcount, maxsize=maxsize) as sink:
        for idx, (png, meta) in enumerate(pairs, 1):
            sample = process_sample(png, meta, image_size, jpeg_quality)
            if sample:
                sink.write(sample)
            if idx % 10_000 == 0:
                logger.info("Processed {}/{} samples", idx, len(pairs))
    logger.success("All shards written to {}", out_dir)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WebDataset shards from PNG/JSON pairs")
    parser.add_argument("--input_dirs", nargs="+", required=True,
                        help="Directories containing PNG/JSON files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where *.tar shards will be written")
    parser.add_argument("--maxcount", type=int, default=100_000,
                        help="Max records per shard (default: 100000)")
    parser.add_argument("--maxsize", type=float, default=20e9,
                        help="Max shard size in bytes (default: 20e9 ≈ 20 GB)")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Resize shortest side to this many pixels (default: 256)")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="JPEG quality (1-100, default: 95)")
    parser.add_argument("--log_level", default="INFO",
                        help="loguru log level (default: INFO)")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    # Clamp quality into Pillow's valid range 1-100.
    jpeg_q = max(1, min(100, args.jpeg_quality))
    if jpeg_q != args.jpeg_quality:
        logger.warning("Clamped --jpeg_quality to {} (valid range is 1-100)", jpeg_q)

    # Configure loguru
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper(),
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    pairs = build_data_paths(args.input_dirs)
    if not pairs:
        logger.error("No PNG/JSON pairs found - exiting")
        sys.exit(1)

    write_shards(
        pairs,
        output_dir=args.output_dir,
        maxcount=args.maxcount,
        maxsize=int(args.maxsize),
        image_size=args.image_size,
        jpeg_quality=jpeg_q,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
