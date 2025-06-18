#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
conda activate ldm

export ACCELERATE_LOG_LEVEL="INFO"
CLIP_MODEL="laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
# CLIP_MODEL="openai/clip-vit-large-patch14"

rm -rf /media/bryan/ssd01/expr/latent_diffusion/debug/run01
accelerate launch --gpu_ids 0, --num_processes 1 ldm/tools/train_latent_diffusion.py \
--output_dir /media/bryan/ssd01/expr/latent_diffusion/debug/run01 \
--pretrained_clip $CLIP_MODEL \
--hf_model_repo_id "CompVis/stable-diffusion-v1-4" \
--hf_model_subfolder "unet" \
--hf_scheduler_repo_id "CompVis/stable-diffusion-v1-4" \
--hf_scheduler_subfolder "scheduler" \
--dataset_tar_specs \
"/media/bryan/nvme2/data/laion2b-aesthetic-square-256/{00000..00001}.tar" \
"/media/bryan/nvme2/data/laion2b-aesthetic-square-plus54-256/00000.tar" \
"/media/bryan/nvme2/data/laion-pop-256/{00000..00004}.tar" \
--train_batch_size 2 \
--gradient_accumulation_steps 2 \
--lr_warmup_steps 500 \
--max_train_steps 1000 \
--checkpointing_steps 100 \
--mixed_precision bf16 \
--dataloader_num_workers 0 \
--noise_offset 0.1 \
--validation_steps 100 \
--validation_prompts \
"A fluffy corgi sitting on a city sidewalk wearing round sunglasses and a tiny leather jacket, golden hour lighting with a soft bokeh background." \
"A woman with long black hair strolling beneath cherry blossoms at night, neon Tokyo glowing in the background, reflections dancing on the river — cinematic and dreamy." \
"A crystal-clear alpine lake surrounded by towering pine trees and snow-dusted mountains under a bright blue sky, reflected symmetrically in the water." \
"A minimalist workspace with a white desk, an iMac, dried pampas grass in a ceramic vase, and sunlight streaming through sheer curtains." \
"A retro-style portrait of a superhero wearing a red and teal suit, illustrated in a 1980s comic book art style with halftone shading and dramatic lighting."