#!/usr/bin/env python3
"""prepare_webdataset.py

Convert PNG/JSON pairs into **WebDataset** shards that contain

* ``<key>.jpg`` - RGB JPEG (quality 95, resized so the shortest side == ``--image_size``)
* ``<key>.json`` - one JSON blob with all metadata

The JSON schema written per sample:

```jsonc
{
  "url": "…",               // from original metadata
  "caption": "…",           // from original metadata
  "sha256": "…",           // from original metadata
  "width": 512,              // resized dimensions
  "height": 768,
  "original_width": 1024,    // dimensions before resize
  "original_height": 1536
}
```

Because WebDataset chooses an encoder based on the *extension* of each sample
field, we bundle every numeric dimension into the JSON payload instead of
keeping standalone keys like ``width``. Passing bare integers would trigger
``ValueError: no handler found for width``.

Usage example
-------------

```bash
python prepare_webdataset.py \
  --input_dirs 00000 00001 \
  --output_dir shards \
  --maxcount 100000 \
  --maxsize 20000000000 \
  --image_size 256
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

__all__ = [
    "build_data_paths",
    "process_sample",
    "write_shards",
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_data_paths(input_dirs: Iterable[str]) -> List[Tuple[Path, Path]]:
    """Return list of (png, json) `Path` pairs discovered under *input_dirs*."""
    pairs: List[Tuple[Path, Path]] = []
    for input_dir in input_dirs:
        for png in sorted(Path(input_dir).rglob("*.png")):
            j = png.with_suffix(".json")
            if j.exists():
                pairs.append((png, j))
            else:
                logger.warning("Missing JSON for {} – skipping", png)
    logger.info("Indexed {} PNG/JSON pairs from {} folders", len(pairs), len(input_dirs))
    return pairs


def resize_image(img: Image.Image, image_size: int) -> Image.Image:
    """Return *img* resized with Lanczos so its shortest side == *image_size*."""
    w, h = img.size
    ratio = image_size / float(min(w, h))
    new_size = (round(w * ratio), round(h * ratio))
    return img.resize(new_size, resample=Image.LANCZOS)


def process_sample(png_path: Path, json_path: Path, image_size: int, jpeg_quality: int = 95) -> dict | None:
    """Convert one PNG/JSON pair into a WebDataset *sample* dict."""
    try:
        # ---------- metadata ------------
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        # ---------- image ---------------
        with Image.open(png_path) as im:
            im = im.convert("RGB")
            orig_w, orig_h = im.size
            im = resize_image(im, image_size)
            resized_w, resized_h = im.size

            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)  # quality 95
            jpg_bytes = buf.getvalue()

        # ---------- merge metadata ------
        meta_out = {
            "url": meta.get("url", ""),
            "caption": meta.get("caption", ""),
            "sha256": meta.get("sha256", ""),
            "width": resized_w,
            "height": resized_h,
            "original_width": orig_w,
            "original_height": orig_h,
        }
        json_bytes = json.dumps(meta_out, ensure_ascii=False).encode("utf-8")

        key = str(meta.get("key", png_path.stem))
        sample = {
            "__key__": key,
            "jpg": jpg_bytes,
            "json": json_bytes,
        }
        return sample
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to process {}: {}", png_path, exc)
        return None


def write_shards(pairs: List[Tuple[Path, Path]], output_dir: str, *,
                 maxcount: int, maxsize: int, image_size: int, jpeg_quality: int = 95) -> None:
    """Write all *pairs* into shard files under *output_dir*."""
    out_dir = Path(output_dir);
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%05d.tar")

    with wds.ShardWriter(pattern, maxcount=maxcount, maxsize=maxsize) as sink:
        for i, (png, js) in enumerate(pairs, 1):
            s = process_sample(png, js, image_size, jpeg_quality=jpeg_quality)
            if s:
                sink.write(s)
            if i % 10_000 == 0:
                logger.info("Processed {}/{} samples", i, len(pairs))
    logger.success("All shards written to {}", out_dir)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare WebDataset shards from PNG/JSON pairs")
    p.add_argument("--input_dirs", nargs="+", required=True,
                   help="One or more directories containing PNG/JSON files")
    p.add_argument("--output_dir", required=True,
                   help="Directory where *.tar shards will be written")
    p.add_argument("--maxcount", type=int, default=100_000,
                   help="Max records per shard (default: 100000)")
    p.add_argument("--maxsize", type=float, default=20e9,
                   help="Max shard size in bytes (default: 20e9 ≈ 20 GB)")
    p.add_argument("--image_size", type=int, default=256,
                   help="Resize shortest side to this many pixels (default: 256)")
    p.add_argument("--log_level", default="INFO",
                   help="Logging level for loguru (default: INFO)")
    p.add_argument("--jpeg_quality", type=int, default=95,
                   help="JPEG quality for saved images (default: 95)")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    a = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level=a.log_level.upper(),
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    pairs = build_data_paths(a.input_dirs)
    if not pairs:
        logger.error("No PNG/JSON pairs found – exiting")
        sys.exit(1)

    write_shards(pairs, a.output_dir,
                 maxcount=a.maxcount,
                 maxsize=int(a.maxsize),
                 image_size=a.image_size,
                 jpeg_quality=a.jpeg_quality)


if __name__ == "__main__":  # pragma: no cover
    main()
