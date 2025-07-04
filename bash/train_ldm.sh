#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
conda activate ldm

export ACCELERATE_LOG_LEVEL="INFO"
CLIP_MODEL="laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
# CLIP_MODEL="openai/clip-vit-large-patch14"

# rm -rf /media/bryan/ssd01/expr/latent_diffusion/debug/run01

DATASET_TAR_SPECS=(
"/media/bryan/nvme2/data/laion2b-aesthetic-square-256/{00000..00001}.tar"
"/media/bryan/nvme2/data/laion2b-aesthetic-square-plus54-256/00000.tar"
"/media/bryan/nvme2/data/laion2b-woman-256/00000.tar"
"/media/bryan/nvme2/data/laion-pop-256/{00000..00004}.tar"
)
VALIDATION_PROMPTS=(
"A happy and energetic corgi standing in a grassy field. The dog has a reddish-brown and white coat, large upright ears, and a short, sturdy build with a fluffy tail curled slightly over its back. Its mouth is open, tongue out, as if smiling or panting. The background is blurred greenery, suggesting a park or natural setting."
"A woman with a fair complexion and light eyes wears subtle makeup that enhances her natural features. Her hair is styled in a low updo, with soft curls framing her face. She has a calm, confident expression and slightly tilted head, adding to her graceful and composed appearance."
"An Asian woman with a fair complexion and light eyes wears subtle makeup that enhances her natural features. Her hair is styled in a low updo, with soft curls framing her face. She has a calm, confident expression and slightly tilted head, adding to her graceful and composed appearance."
"A man is rock climbing a steep vertical cliff face, high above a forested valley with dramatic mountain scenery in the background. He wears a red t-shirt, gray climbing pants, and a harness with gear clipped to it. His posture shows focus and determination as he reaches upward with one arm. The bright daylight and clear sky add to the sense of height and exposure."
"A stunning lakeside scene features crystal-clear turquoise water in the foreground, revealing smooth stones beneath the surface. Pine trees and large boulders line the shore, their reflections mirrored on the water. In the distance, snow-capped mountains rise beneath a vibrant blue sky scattered with soft, wispy clouds. The setting is serene, evoking a sense of natural tranquility."
"A minimalist workspace features a sleek monitor displaying a colorful macOS wallpaper, mounted above a wooden desk. A silver Mac mini stands vertically beside it. The setup includes a white Apple keyboard, wireless mouse on a dark desk mat, a black notebook, and two small potted plants. A candle adds warmth, with natural light streaming through a nearby window."
"Three men stand outdoors laughing joyfully, captured in a moment of genuine camaraderie. One wears a beige coat, another a patterned red coat, and the third a brown coat. Their body language shows close friendship, with arms around each other. The background features trees and soft natural light, enhancing the warm and cheerful atmosphere."
"A delicious cheeseburger stacked with two juicy beef patties, melted cheddar cheese, caramelized onions, and creamy sauce under a soft golden bun. Fresh lettuce, a thick tomato slice, and pickles sit beneath the patties, all layered on a toasted bottom bun. The burger is presented against a plain white background, highlighting its fresh and savory ingredients."
"A pencil drawing of a woman with expressive eyes and softly parted lips, gently biting a stem or strand. Her hair falls in loose waves around her face, and one hand delicately holds a flower near her chin. Detailed shading and fine lines create a realistic, lifelike texture. The artwork is signed and dated in the bottom corner."
)

# accelerate launch --gpu_ids 0, --num_processes 1 ldm/tools/train_latent_diffusion.py \
# --output_dir /media/bryan/ssd01/expr/latent_diffusion/bs84_500k/run01 \
# --pretrained_clip $CLIP_MODEL \
# --hf_model_repo_id "CompVis/stable-diffusion-v1-4" \
# --hf_model_subfolder "unet" \
# --hf_scheduler_repo_id "CompVis/stable-diffusion-v1-4" \
# --hf_scheduler_subfolder "scheduler" \
# --dataset_tar_specs "${DATASET_TAR_SPECS[@]}" \
# --train_batch_size 42 \
# --gradient_accumulation_steps 2 \
# --lr_warmup_steps 10000 \
# --max_train_steps 500000 \
# --checkpointing_steps 5000 \
# --mixed_precision bf16 \
# --dataloader_num_workers 4 \
# --noise_offset 0.1 \
# --validation_steps 5000 \
# --validation_prompts "${VALIDATION_PROMPTS[@]}"


# rm -rf /media/bryan/ssd01/expr/latent_diffusion/debug/run02
# accelerate launch --gpu_ids 0, --num_processes 1 ldm/tools/train_latent_diffusion.py \
# --output_dir /media/bryan/ssd01/expr/latent_diffusion/debug/run02 \
# --pretrained_clip $CLIP_MODEL \
# --hf_model_repo_id "CompVis/stable-diffusion-v1-4" \
# --hf_model_subfolder "unet" \
# --hf_scheduler_repo_id "CompVis/stable-diffusion-v1-4" \
# --hf_scheduler_subfolder "scheduler" \
# --dataset_tar_specs "${DATASET_TAR_SPECS[@]}" \
# --train_batch_size 5 \
# --gradient_accumulation_steps 2 \
# --lr_warmup_steps 10 \
# --max_train_steps 10 \
# --checkpointing_steps 100 \
# --mixed_precision bf16 \
# --dataloader_num_workers 4 \
# --noise_offset 0.1 \
# --validation_steps 10 \
# --validation_prompts "${VALIDATION_PROMPTS[@]}" \
# --generations_per_val_prompt 3 \
# --text_conditioning_dropout 0.1 \
# --seed 42

# --output_dir /media/bryan/ssd01/expr/latent_diffusion/hf_bs252_200k \

# export TORCH_LOGS="all"
# accelerate launch --gpu_ids 0, --num_processes 1 ldm/tools/train_latent_diffusion.py \
# --output_dir /media/bryan/ssd01/expr/latent_diffusion/hf_bs250_200k \
# --pretrained_clip $CLIP_MODEL \
# --hf_model_repo_id "CompVis/stable-diffusion-v1-4" \
# --hf_model_subfolder "unet" \
# --hf_scheduler_repo_id "CompVis/stable-diffusion-v1-4" \
# --hf_scheduler_subfolder "scheduler" \
# --dataset_tar_specs "${DATASET_TAR_SPECS[@]}" \
# --enable_torch_compile --use_8bit_adam \
# --train_batch_size 50 \
# --gradient_accumulation_steps 5 \
# --lr_warmup_steps 10000 \
# --max_train_steps 200000 \
# --checkpointing_steps 1000 \
# --mixed_precision bf16 \
# --dataloader_num_workers 2 \
# --noise_offset 0.1 \
# --validation_steps 1000 \
# --validation_prompts "${VALIDATION_PROMPTS[@]}" \
# --generations_per_val_prompt 3 \
# --text_conditioning_dropout 0.1 \
# --seed 42 \
# --resume_from_checkpoint latest


# DATASET_TAR_SPECS=(
# "/media/bryan/nvme2/data/laion2b-aesthetic-square-256/{00000..00001}.tar"
# "/media/bryan/nvme2/data/laion2b-aesthetic-square-plus54-256/00000.tar"
# "/media/bryan/nvme2/data/laion-pop-256/{00000..00004}.tar"
# )
# CAPTION_FILTERS=(
# "an outdoor scene in nature, like a beach, forest, mountain, river, or grassy field"
# )
# VALIDATION_PROMPTS=(
# "A stunning lakeside scene features crystal-clear turquoise water in the foreground, revealing smooth stones beneath the surface. Pine trees and large boulders line the shore, their reflections mirrored on the water. In the distance, snow-capped mountains rise beneath a vibrant blue sky scattered with soft, wispy clouds. The setting is serene, evoking a sense of natural tranquility."
# )
# export TORCH_LOGS="all"
# rm -rf /media/bryan/ssd01/expr/latent_diffusion/train_nature/ldm_bs250_100k
# accelerate launch --gpu_ids 0, --num_processes 1 ldm/tools/train_latent_diffusion.py \
# --output_dir /media/bryan/ssd01/expr/latent_diffusion/train_nature/ldm_bs250_100k \
# --pretrained_clip $CLIP_MODEL \
# --hf_model_repo_id "CompVis/stable-diffusion-v1-4" \
# --hf_model_subfolder "unet" \
# --hf_scheduler_repo_id "CompVis/stable-diffusion-v1-4" \
# --hf_scheduler_subfolder "scheduler" \
# --dataset_tar_specs "${DATASET_TAR_SPECS[@]}" \
# --enable_torch_compile --use_8bit_adam \
# --train_batch_size 50 \
# --gradient_accumulation_steps 5 \
# --lr_warmup_steps 10000 \
# --max_train_steps 200000 \
# --checkpointing_steps 1000 \
# --mixed_precision bf16 \
# --dataloader_num_workers 2 \
# --noise_offset 0.1 \
# --validation_steps 1000 \
# --validation_prompts "${VALIDATION_PROMPTS[@]}" \
# --generations_per_val_prompt 3 \
# --text_conditioning_dropout 0.1 \
# --seed 42 \
# --caption_filters "${CAPTION_FILTERS[@]}" \
# --caption_filter_thresholds 0.4 \
# --sentence_transformer_batch_size 1000


VALIDATION_PROMPTS=(
"A woman with a fair complexion and light eyes wears subtle makeup that enhances her natural features. Her hair is styled in a low updo, with soft curls framing her face. She has a calm, confident expression and slightly tilted head, adding to her graceful and composed appearance."
"A stunning lakeside scene features crystal-clear turquoise water in the foreground, revealing smooth stones beneath the surface. Pine trees and large boulders line the shore, their reflections mirrored on the water. In the distance, snow-capped mountains rise beneath a vibrant blue sky scattered with soft, wispy clouds. The setting is serene, evoking a sense of natural tranquility."
"A misty pine forest at sunrise, with golden light piercing through the trees and a deer grazing near a moss-covered rock."
"A narrow desert canyon at noon, red sandstone walls casting sharp shadows and a lone hawk soaring overhead."
"A minimalist workspace features a sleek monitor displaying a colorful macOS wallpaper, mounted above a wooden desk. A silver Mac mini stands vertically beside it. The setup includes a white Apple keyboard, wireless mouse on a dark desk mat, a black notebook, and two small potted plants. A candle adds warmth, with natural light streaming through a nearby window."
"A two-story suburban house with a red door, kids playing on the front lawn, and mailboxes lining a quiet residential street."
"A luxurious living room with marble floors, a crystal chandelier, velvet furniture, and tall windows revealing a garden view."
"A sleek black sports car parked under neon lights at night, its polished body reflecting city skyscrapers in the background."
"A delicious cheeseburger stacked with two juicy beef patties, melted cheddar cheese, caramelized onions, and creamy sauce under a soft golden bun. Fresh lettuce, a thick tomato slice, and pickles sit beneath the patties, all layered on a toasted bottom bun. The burger is presented against a plain white background, highlighting its fresh and savory ingredients."
)

export TORCH_LOGS="all"
accelerate launch --gpu_ids 0, --num_processes 1 ldm/tools/train_latent_diffusion.py \
--output_dir /media/bryan/ssd01/expr/latent_diffusion/custom_UNet_bs250_cosine_100k \
--pretrained_clip $CLIP_MODEL \
--hf_scheduler_repo_id "CompVis/stable-diffusion-v1-4" \
--hf_scheduler_subfolder "scheduler" \
--dataset_tar_specs "${DATASET_TAR_SPECS[@]}" \
--enable_torch_compile --use_8bit_adam \
--train_batch_size 50 \
--gradient_accumulation_steps 5 \
--lr_warmup_steps 10000 \
--max_train_steps 100000 \
--lr_constant_steps 60000 \
--lr_scheduler cosine_with_warmup_then_constant \
--checkpointing_steps 1000 \
--milestone_checkpoints 10000 50000 70000 \
--mixed_precision bf16 \
--dataloader_num_workers 2 \
--noise_offset 0.1 \
--validation_steps 1000 \
--validation_prompts "${VALIDATION_PROMPTS[@]}" \
--generations_per_val_prompt 3 \
--text_conditioning_dropout 0.1 \
--seed 42 \
--resume_from_checkpoint latest