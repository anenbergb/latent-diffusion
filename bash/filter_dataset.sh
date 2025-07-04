#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
conda activate ldm

DATASET_TAR_SPECS=(
"/media/bryan/nvme2/data/laion-pop-256/{00000..00004}.tar"
"/media/bryan/nvme2/data/laion2b-aesthetic-square-256/{00000..00001}.tar"
"/media/bryan/nvme2/data/laion2b-aesthetic-square-plus54-256/00000.tar"
"/media/bryan/nvme2/data/laion2b-woman-256/00000.tar"
)

# "a photo of a forest" \
# "a photo of a mountain" \
# "a photo of the beach" \
# "a photo taken in nature" \
# "an outdoor landscape scene" \

python ldm/tools/filter_dataset.py \
--device cuda \
--dataset_tar_specs "${DATASET_TAR_SPECS[@]}" \
--caption_filters \
"an outdoor scene in nature, like a beach, forest, mountain, river, or grassy field" \
--caption_filter_thresholds 0.4 \
--save_sample_images 100 \
--output_dir /media/bryan/ssd01/expr/latent_diffusion/filter_dataset/nature