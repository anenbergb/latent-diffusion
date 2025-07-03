import argparse
import logging
import os
from typing import Optional, List

import webdataset as wds
from braceexpand import braceexpand
import sentence_transformers
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def parse_args():
    parser = argparse.ArgumentParser(
        "filter_dataset",
        description="Filter and count dataset samples based on caption similarity using SentenceTransformers",
    )
    parser.add_argument(
        "--dataset_tar_specs",
        type=str,
        nargs="+",
        required=True,
        help="Brace-expandable TAR shard specifications for WebDataset. "
        "Examples: '/data/laion/{00000..09999}.tar' or '/data/extra/00000.tar'. "
        "Multiple specs can be provided and will be expanded and combined.",
    )
    parser.add_argument(
        "--caption_filters",
        type=str,
        nargs="+",
        default=None,
        help="List of text phrases to use as filters for dataset captions. "
        "Samples with captions similar to any of these phrases (above the threshold) will be kept. "
        "The filters are combined using logical OR - a sample passes if it matches ANY filter. "
        "Example: --caption_filters 'a cat' 'dog playing' 'beautiful landscape'",
    )
    parser.add_argument(
        "--caption_filter_thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Cosine similarity thresholds (0.0-1.0) for each caption filter. "
        "Must provide one threshold per filter, or none to use default 0.5 for all. "
        "Higher values = stricter filtering (more similar to filter text required). "
        "Typical values: 0.3-0.7. Example: --caption_filter_thresholds 0.4 0.6 0.5",
    )
    parser.add_argument(
        "--sentence_transformer_model_name",
        type=str,
        default="paraphrase-MiniLM-L6-v2",
        help="HuggingFace SentenceTransformer model name for computing text embeddings. "
        "Popular options: 'paraphrase-MiniLM-L6-v2' (fast, 384-dim), "
        "'all-MiniLM-L6-v2' (balanced), 'all-mpnet-base-v2' (best quality, slower). "
        "See https://huggingface.co/sentence-transformers for more models.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.caption_filters and args.caption_filter_thresholds:
        if len(args.caption_filters) != len(args.caption_filter_thresholds):
            parser.error(
                f"Number of caption filters ({len(args.caption_filters)}) must match "
                f"number of thresholds ({len(args.caption_filter_thresholds)})"
            )

    return args


class CaptionFilter:
    """
    Filter dataset captions based on similarity to reference texts using SentenceTransformers.

    Uses cosine similarity between embeddings to determine if a caption should be kept.
    Supports multiple filter texts with individual thresholds (OR logic).

    Default model 'paraphrase-MiniLM-L6-v2' provides fast, ~384-dimensional embeddings.
    """

    def __init__(
        self,
        filter_texts: List[str],
        filter_thresholds: Optional[List[float]] = None,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """
        Initialize the caption filter.

        Args:
            filter_texts: List of reference texts to filter against
            filter_thresholds: Cosine similarity thresholds for each filter text
            model_name: SentenceTransformer model name
            device: Device to run the model on ('cpu' or 'cuda')
        """
        logging.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.filter_texts = filter_texts
        self.thresholds = filter_thresholds or [0.5] * len(filter_texts)

        if len(self.thresholds) != len(filter_texts):
            raise ValueError(
                f"Number of thresholds ({len(self.thresholds)}) must match number of filter texts ({len(filter_texts)})"
            )

        logging.info(f"Computing embeddings for {len(filter_texts)} filter texts...")
        self.filter_embeddings = [self.model.encode(text, convert_to_tensor=True) for text in filter_texts]

        # Log filter configuration
        for i, (text, threshold) in enumerate(zip(filter_texts, self.thresholds)):
            logging.info(f"Filter {i + 1}: '{text}' (threshold: {threshold:.3f})")

    def __call__(self, caption: str) -> bool:
        """
        Check if a caption passes any of the filters.

        Args:
            caption: Caption text to check

        Returns:
            True if caption passes any filter, False otherwise
        """
        caption_embedding = self.model.encode(caption, convert_to_tensor=True)

        for i, (filter_embedding, threshold) in enumerate(zip(self.filter_embeddings, self.thresholds)):
            score = sentence_transformers.util.cos_sim(filter_embedding, caption_embedding).item()
            if score >= threshold:
                logging.debug(f"Caption passed filter {i + 1} with score {score:.3f}: '{caption[:50]}...'")
                return True

        return False


def main():
    """Main function to filter and count dataset samples."""
    args = parse_args()

    logging.info("Starting dataset filtering process...")

    # Expand TAR file specifications
    logging.info("Expanding TAR file specifications...")
    tar_files = []
    for spec in args.dataset_tar_specs:
        expanded = braceexpand(spec)
        tar_files.extend(expanded)
        logging.info(f"Spec '{spec}' expanded to {len(expanded)} files")

    if not tar_files:
        raise FileNotFoundError("No tar files resolved from --dataset_tar_specs")

    logging.info(f"Total TAR files to process: {len(tar_files)}")

    # Create WebDataset
    logging.info("Creating WebDataset...")
    dataset = wds.WebDataset(
        tar_files,
        repeat=False,
        shardshuffle=False,
        detshuffle=False,
    ).to_tuple("jpg", "json")

    # Initialize caption filter
    if args.caption_filters:
        caption_filter = CaptionFilter(
            args.caption_filters,
            args.caption_filter_thresholds,
            model_name=args.sentence_transformer_model_name,
        )
        logging.info("Caption filter initialized")
    else:
        caption_filter = None
        logging.info("No caption filters specified - counting all samples")

    # Process dataset
    logging.info("Processing dataset samples...")
    dataset_iter = iter(dataset)
    total_count = 0
    filtered_count = 0

    try:
        for image_data, json_data in dataset_iter:
            total_count += 1

            if caption_filter is None:
                # No filtering - count all samples
                filtered_count += 1
            else:
                # Apply caption filter
                caption = json_data.get("caption", "")
                if caption and caption_filter(caption):
                    filtered_count += 1

            # Log progress every 1000 samples
            if total_count % 1000 == 0:
                pass_rate = (filtered_count / total_count) * 100 if total_count > 0 else 0
                logging.info(f"Processed {total_count:,} samples, {filtered_count:,} passed filters ({pass_rate:.1f}%)")

    except KeyboardInterrupt:
        logging.warning("Processing interrupted by user")
    except Exception as e:
        logging.error(f"Error processing dataset: {e}")
        raise

    # Final statistics
    pass_rate = (filtered_count / total_count) * 100 if total_count > 0 else 0

    logging.info("=" * 60)
    logging.info("DATASET FILTERING RESULTS")
    logging.info("=" * 60)
    logging.info(f"Total samples processed:     {total_count:,}")
    logging.info(f"Samples passing filters:     {filtered_count:,}")
    logging.info(f"Pass rate:                   {pass_rate:.2f}%")
    logging.info(f"Samples filtered out:        {total_count - filtered_count:,}")

    if caption_filter:
        logging.info(f"Filter texts used:           {len(args.caption_filters)}")
        for i, text in enumerate(args.caption_filters):
            threshold = args.caption_filter_thresholds[i] if args.caption_filter_thresholds else 0.5
            logging.info(f"  Filter {i + 1}: '{text}' (threshold: {threshold:.3f})")

    logging.info("=" * 60)


if __name__ == "__main__":
    main()
