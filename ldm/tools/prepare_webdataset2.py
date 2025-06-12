#!/usr/bin/env python3
"""prepare_webdataset.py  —  *tgz ➜ WebDataset*

Transform an **existing WebDataset** (PNG + JSON inside *.tar* shards) into a
new WebDataset of **resized JPEGs** whose metadata are augmented with captions
and NSFW flags fetched from a Hugging Face dataset.

Pipeline
~~~~~~~~
1. **Input shards**: supplied via `--input_tars`, e.g.
   `/data/laion-pop/{00000..00058}.tar`.
2. **Caption/NSFW lookup**: an HF dataset (default ``laion/laion-pop``
   split ``train``) is indexed by *url* → {caption, nsfw_prediction}.
3. Iterate the source `WebDataset`, resize each PNG to ``--image_size`` on its
   shortest side, encode as JPEG (``--jpeg_quality``), merge metadata.
4. Write to a `wds.ShardWriter` under ``--output_dir`` with optional shard size
   and count limits.

Resulting sample inside each new *.tar*:
```text
<key>.jpg   # RGB, quality N
<key>.json  # {
            #   "url": …,
            #   "caption": …,
            #   "nsfw_prediction": …,
            #   "sha256": …,
            #   "width": …,
            #   "height": …,
            #   "original_width": …,
            #   "original_height": …
            # }
```

Dependencies
------------
```bash
pip install Pillow webdataset loguru datasets
```
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

from PIL import Image
from loguru import logger
import webdataset as wds
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def strip_problematic_metadata(img: Image.Image) -> None:  # noqa: D401
    """Remove XMP/XML payloads that exceed JPEG segment limits."""
    for k in ("xmp", "xml"):
        img.info.pop(k, None)


def resize_image(img: Image.Image, target: int) -> Image.Image:  # noqa: D401
    """Lanczos‑resize *img* so its shortest side = *target*."""
    w, h = img.size
    scale = target / float(min(w, h))
    return img.resize((round(w * scale), round(h * scale)), resample=Image.LANCZOS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_url_index(hf_name: str, hf_split: str, caption_key: str) -> Dict[str, dict]:
    """Return *url → {'caption':…, 'nsfw_prediction':…}* dictionary."""
    logger.info("Loading HF dataset {} | split={} …", hf_name, hf_split)
    ds = load_dataset(hf_name, split=hf_split, streaming=False)  # full load → memory OK (≈580k)
    logger.info("Loaded {} rows; indexing by URL", len(ds))

    idx: Dict[str, dict] = {}
    missing_caption = 0
    for row in ds:
        url = row["url"]
        if not url:
            continue
        caption = row.get(caption_key) or row.get("llava_caption") or ""
        if caption == "":
            missing_caption += 1
        idx[url] = {
            "caption": caption,
            "nsfw_prediction": row.get("nsfw_prediction"),
        }
    if missing_caption:
        logger.warning("{} rows lacked '{}' caption", missing_caption, caption_key)
    logger.success("URL index built with {} entries", len(idx))
    return idx


# ---------------------------------------------------------------------------
# Sample processing
# ---------------------------------------------------------------------------

def process_sample(
    sample: dict,
    url_index: Dict[str, dict],
    *,
    image_size: int,
    jpeg_quality: int,
) -> dict | None:
    """Transform one source sample into target WebDataset sample dict."""
    try:
        # ---------- metadata from source JSON ----------
        meta_src = json.loads(sample["json"].decode("utf-8"))
        url = meta_src.get("url", "")
        sha256 = meta_src.get("sha256", "")
        o_w = meta_src.get("original_width") or meta_src.get("width")
        o_h = meta_src.get("original_height") or meta_src.get("height")

        # ---------- caption / nsfw from HF index ----------
        extra = url_index.get(url, {})
        if not extra:
            logger.debug("URL not found in HF index: {}", url)
        caption = extra.get("caption", "")
        nsfw_pred = extra.get("nsfw_prediction")

        # ---------- image processing ----------
        with Image.open(io.BytesIO(sample["png"])) as img:
            img = img.convert("RGB")
            img_resized = resize_image(img, image_size)
            w_r, h_r = img_resized.size
            strip_problematic_metadata(img_resized)
            buf = io.BytesIO()
            img_resized.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            jpg_bytes = buf.getvalue()

        meta_out = {
            "url": url,
            "caption": caption,
            "nsfw_prediction": nsfw_pred,
            "sha256": sha256,
            "width": w_r,
            "height": h_r,
            "original_width": o_w,
            "original_height": o_h,
        }
        json_bytes = json.dumps(meta_out, ensure_ascii=False).encode("utf-8")

        return {
            "__key__": str(sample.get("__key__")),
            "jpg": jpg_bytes,
            "json": json_bytes,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to process sample {}: {}", sample.get("__key__"), exc)
        return None


# ---------------------------------------------------------------------------
# Main write loop
# ---------------------------------------------------------------------------

def convert_dataset(
    input_tars: str,
    output_dir: str,
    *,
    url_index: Dict[str, dict],
    maxcount: int,
    maxsize: int,
    image_size: int,
    jpeg_quality: int,
) -> None:
    """Stream the source shards and write the transformed dataset."""
    logger.info("Opening source WebDataset with pattern: {}", input_tars)
    src_ds = wds.WebDataset(input_tars, handler=wds.ignore_and_continue)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%05d.tar")

    with wds.ShardWriter(pattern, maxcount=maxcount, maxsize=maxsize) as sink:
        for idx, sample in enumerate(src_ds, 1):
            tgt = process_sample(sample, url_index, image_size=image_size, jpeg_quality=jpeg_quality)
            if tgt:
                sink.write(tgt)
            if idx % 10_000 == 0:
                logger.info("Processed {:,} samples", idx)
    logger.success("Finished. Shards written to {}", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser(description="Convert WebDataset to resized JPEG variant with HF captions")
    p.add_argument("--input_tars", required=True,
                   help="Input tar(s) glob, e.g. /data/old/{00000..00010}.tar")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write new *.tar shards")
    p.add_argument("--hf_dataset_name", default="laion/laion-pop",
                   help="HF dataset providing captions (default: laion/laion-pop)")
    p.add_argument("--hf_dataset_split", default="train",
                   help="Split to load (default: train)")
    p.add_argument("--caption_key", default="cogvlm_caption",
                   help="Which caption column to use (default: cogvlm_caption)")
    p.add_argument("--image_size", type=int, default=256,
                   help="Resize shortest side to this many pixels (default: 256)")
    p.add_argument("--jpeg_quality", type=int, default=95,
                   help="JPEG quality (1-100, default: 95)")
    p.add_argument("--maxcount", type=int, default=100_000,
                   help="Max records per shard (default: 100_000)")
    p.add_argument("--maxsize", type=float, default=20e9,
                   help="Max shard size in bytes (default: 20 GB)")
    p.add_argument("--log_level", default="INFO",
                   help="loguru logging level (default: INFO)")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:  # noqa: D401
    args = parse_args(argv)

    # Configure logging first so subsequent helpers inherit sink
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper(),
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    jpeg_q = max(1, min(100, args.jpeg_quality))
    if jpeg_q != args.jpeg_quality:
        logger.warning("Clamped --jpeg_quality to {} (1-100)", jpeg_q)

    url_index = build_url_index(args.hf_dataset_name, args.hf_dataset_split, args.caption_key)

    convert_dataset(
        input_tars=args.input_tars,
        output_dir=args.output_dir,
        url_index=url_index,
        maxcount=args.maxcount,
        maxsize=int(args.maxsize),
        image_size=args.image_size,
        jpeg_quality=jpeg_q,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
