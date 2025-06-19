#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
conda activate ldm

export ACCELERATE_LOG_LEVEL="INFO"
CLIP_MODEL="laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
# CLIP_MODEL="openai/clip-vit-large-patch14"

# rm -rf /media/bryan/ssd01/expr/latent_diffusion/debug/run01
accelerate launch --gpu_ids 0, --num_processes 1 ldm/tools/train_latent_diffusion.py \
--output_dir /media/bryan/ssd01/expr/latent_diffusion/bs84_500k/run01 \
--pretrained_clip $CLIP_MODEL \
--hf_model_repo_id "CompVis/stable-diffusion-v1-4" \
--hf_model_subfolder "unet" \
--hf_scheduler_repo_id "CompVis/stable-diffusion-v1-4" \
--hf_scheduler_subfolder "scheduler" \
--dataset_tar_specs \
"/media/bryan/nvme2/data/laion2b-aesthetic-square-256/{00000..00001}.tar" \
"/media/bryan/nvme2/data/laion2b-aesthetic-square-plus54-256/00000.tar" \
"/media/bryan/nvme2/data/laion2b-woman-256/00000.tar" \
"/media/bryan/nvme2/data/laion-pop-256/{00000..00004}.tar" \
--train_batch_size 42 \
--gradient_accumulation_steps 2 \
--lr_warmup_steps 10000 \
--max_train_steps 500000 \
--checkpointing_steps 5000 \
--mixed_precision bf16 \
--dataloader_num_workers 4 \
--noise_offset 0.1 \
--validation_steps 5000 \
--validation_prompts \
"A happy and energetic corgi standing in a grassy field. The dog has a reddish-brown and white coat, large upright ears, and a short, sturdy build with a fluffy tail curled slightly over its back. Its mouth is open, tongue out, as if smiling or panting. The background is blurred greenery, suggesting a park or natural setting." \
"A woman with a fair complexion and light eyes wears subtle makeup that enhances her natural features. Her hair is styled in a low updo, with soft curls framing her face. She has a calm, confident expression and slightly tilted head, adding to her graceful and composed appearance." \
"An Asian woman with a fair complexion and light eyes wears subtle makeup that enhances her natural features. Her hair is styled in a low updo, with soft curls framing her face. She has a calm, confident expression and slightly tilted head, adding to her graceful and composed appearance." \
"A man is rock climbing a steep vertical cliff face, high above a forested valley with dramatic mountain scenery in the background. He wears a red t-shirt, gray climbing pants, and a harness with gear clipped to it. His posture shows focus and determination as he reaches upward with one arm. The bright daylight and clear sky add to the sense of height and exposure." \
"A stunning lakeside scene features crystal-clear turquoise water in the foreground, revealing smooth stones beneath the surface. Pine trees and large boulders line the shore, their reflections mirrored on the water. In the distance, snow-capped mountains rise beneath a vibrant blue sky scattered with soft, wispy clouds. The setting is serene, evoking a sense of natural tranquility." \
"A minimalist workspace features a sleek monitor displaying a colorful macOS wallpaper, mounted above a wooden desk. A silver Mac mini stands vertically beside it. The setup includes a white Apple keyboard, wireless mouse on a dark desk mat, a black notebook, and two small potted plants. A candle adds warmth, with natural light streaming through a nearby window." \
"Three men stand outdoors laughing joyfully, captured in a moment of genuine camaraderie. One wears a beige coat, another a patterned red coat, and the third a brown coat. Their body language shows close friendship, with arms around each other. The background features trees and soft natural light, enhancing the warm and cheerful atmosphere." \
"A delicious cheeseburger stacked with two juicy beef patties, melted cheddar cheese, caramelized onions, and creamy sauce under a soft golden bun. Fresh lettuce, a thick tomato slice, and pickles sit beneath the patties, all layered on a toasted bottom bun. The burger is presented against a plain white background, highlighting its fresh and savory ingredients." \
"A pencil drawing of a woman with expressive eyes and softly parted lips, gently biting a stem or strand. Her hair falls in loose waves around her face, and one hand delicately holds a flower near her chin. Detailed shading and fine lines create a realistic, lifelike texture. The artwork is signed and dated in the bottom corner."