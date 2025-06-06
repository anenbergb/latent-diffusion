#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
conda activate ldm

accelerate launch ldm/tools/hf_train_autoencoderkl.py \
--pretrained_model_name_or_path stabilityai/sd-vae-ft-mse \
--output_dir /home/bryan/expr/latent_diffusion/hf_autoencoder/pretrained01 \
--train_batch_size 6 \
--gradient_accumulation_steps 5 \
--checkpoints_total_limit 2 \
--num_train_epochs 1 \
--checkpointing_steps 500 \
--learning_rate 4.5e-6 \
--lr_scheduler cosine \
--lr_warmup_steps 500 \
--use_8bit_adam \
--use_ema \
--dataloader_num_workers 4 \
--report_to tensorboard \
--mixed_precision bf16 \
--enable_xformers_memory_efficient_attention \
--dataset_name ILSVRC/imagenet-1k \
--validation_steps 500

# --max_train_samples 10000 \
