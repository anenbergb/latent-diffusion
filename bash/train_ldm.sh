#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
conda activate ldm

export ACCELERATE_LOG_LEVEL="INFO"

accelerate launch --gpu_ids 0, --num_processes 1 ldm/tools/train_latent_diffusion.py \
--output_dir /media/bryan/ssd01/expr/latent_diffusion/debug/run01 \
--pretrained_denoiser "CompVis/stable-diffusion-v1-4" \
--dataset_tar_specs \
"/media/bryan/nvme2/data/laion2b-aesthetic-square-256/{00000..00001}.tar" \
"/media/bryan/nvme2/data/laion2b-aesthetic-square-plus54-256/00000.tar" \
"/media/bryan/nvme2/data/laion-pop-256/{00000..00004}.tar" \
--train_batch_size 5 \
--gradient_accumulation_steps 2 \
--lr_warmup_steps 500 \
--max_train_steps 1000 \
--checkpointing_steps 100 \
--mixed_precision bf16 \
--dataloader_num_workers 0 \
--noise_offset 0.1