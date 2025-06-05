import fiftyone.zoo as foz

# Define the splits and label types you want
splits = ["train", "validation", "test"]
label_types = ["classifications"]

# Set this to None to download all images
max_samples = None

# Download each split with all label types
for split in splits:
    print(f"Downloading {split} split...")
    foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=label_types,
        max_samples=max_samples,  # Download all
        dataset_name=f"open-images-v7-{split}"
    )
