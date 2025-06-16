import argparse
import os
import random
import math
from pathlib import Path
from contextlib import nullcontext
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm
import logging
from braceexpand import braceexpand
import webdataset as wds

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import make_image_grid
from transformers import CLIPTextModel, CLIPTokenizer


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser("Stable-Diffusion WebDataset finetuner")

    # --- model & data ------------------------------------------------------- #
    parser.add_argument(
        "--pretrained_clip",
        type=str,
        default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        help="HF path to pretrained CLIP model (text encoder). Options could include "
        "'openai/clip-vit-large-patch14', 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K', 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'",
    )
    parser.add_argument(
        "--pretrained_vae",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="HF path to pretrained VAE model (image encoder/decoder). Options could include "
        "'stabilityai/sd-vae-ft-mse' which was used in stable diffusion v1.4 and works with input resolutions or 256x256 and 512x512.",
    )
    parser.add_argument(
        "--pretrained_denoiser",
        type=str,
        required=True,
        help="HF path or local path to pretrained latent denoising model, e.g. the U-Net or DiT. "
        "For stable diffusion use 'CompVis/stable-diffusion-v1-4'",
    )
    parser.add_argument(
        "--dataset_tar_specs",
        type=str,
        nargs="+",
        required=True,
        help="Brace-expandable TAR shard specs (e.g. '/data/laion/{00000..09999}.tar /data/extra/00000.tar').",
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=10000,
        help="Number of samples in WebDataset shuffle buffer.",
    )

    # --- image & caption preprocessing ------------------------------------- #
    parser.add_argument("--resolution", type=int, default=256, help="Square image size for training.")
    parser.add_argument(
        "--random_flip", type=float, default=0.5, help="Probability to apply horizontal flip augmentation."
    )
    parser.add_argument("--resize", action="store_true", help="Resize the input to the target image")

    # --- training hyper-params --------------------------------------------- #
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size per step.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Grad-accum steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Base LR.")
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale LR by (batch x grad_accum). Useful for linear-scaling rule.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="LR scheduler type.",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Scheduler warm-up steps.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        required=True,
        help="Total optimization steps to perform (mandatory with WebDataset).",
    )

    # --- advanced features -------------------------------------------------- #
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma (paper 2303.09556).")
    parser.add_argument("--dream_training", action="store_true", help="Use DREAM loss (paper 2312.00210).")
    parser.add_argument(
        "--dream_detail_preservation",
        type=float,
        default=1.0,
        help="DREAM detail-preservation factor p (>0).",
    )
    parser.add_argument("--noise_offset", type=float, default=0.0, help="Add offset noise (CrossLabs blog).")
    parser.add_argument("--input_perturbation", type=float, default=0.0, help="Extra noise perturbation factor.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use bitsandbytes 8-bit AdamW.")
    parser.add_argument("--use_ema", action="store_true", help="Maintain EMA of UNet params.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use foreach-based EMA update (faster).")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Enable xformers.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=("epsilon", "v_prediction"),
        help="The prediction_type that shall be used for training." \
        "Choose between 'epsilon' or 'v_prediction'"
    )

    # --- misc / logging / checkpoints -------------------------------------- #
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save every N steps.")
    parser.add_argument(
        "--checkpoint_total_limit",
        type=int,
        default=3,
        help="Max checkpoints to keep (handled by Accelerate).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=10000,
        help="Run validation every N optimization steps.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        nargs="+",
        default=None,
        help="Prompts to visualize during validation.",
    )
    parser.add_argument("--mixed_precision", choices=["fp32", "fp16", "bf16"], default="bf16", help="AMP dtype.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Worker processes for DataLoader.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned", help="Checkpoint / TB dir.")

    args = parser.parse_args()
    return args


# --------------------------------------------------------------------------- #
# Data utilities
# --------------------------------------------------------------------------- #
def build_webdataset(args, tokenizer):
    # Expand brace patterns into concrete *.tar paths
    tar_files = []
    for spec in args.dataset_tar_specs:
        tar_files.extend(braceexpand(spec))
    if not tar_files:
        raise FileNotFoundError("No tar files resolved from --dataset_tar_specs")

    # WebDataset decode logic
    # https://github.com/webdataset/webdataset/blob/main/webdataset/autodecode.py#L299
    dataset = (
        wds.WebDataset(
            tar_files,
            repeat=False,
            shardshuffle=len(tar_files),
            detshuffle=True,
            seed=args.seed,
        )
        .shuffle(args.shuffle_buffer)
        .decode("pilrgb")
        .to_tuple("jpg", "json")
    )

    interpolation = transforms.InterpolationMode.LANCZOS
    ops = [
        transforms.ToImage(),  # Convert PIL image to torchvision.tv_tensors.Image
        transforms.ToDtype(torch.uint8, scale=True),
    ]
    if args.resize:
        ops.append(transforms.Resize(args.resolution, interpolation=interpolation))
    ops.extend(
        [
            transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip(p=args.random_flip),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5], [0.5]),  # hard-coded to mean=std=0.5 due to VAE
        ]
    )
    tx = transforms.Compose(ops)

    def _map(sample):
        # input longer than 'max_model_length' (77) get truncated
        # input shorter than 'max_model_length' get padded
        img, meta = sample
        caption = meta.get("caption", "An image.")
        return {
            "pixel_values": tx(img),
            "input_ids": tokenizer(
                caption,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0),
        }

    return dataset.map(_map)


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


class DataloaderMaxSteps:
    def __init__(self, dataloader: torch.utils.data.DataLoader, max_steps: int, start_step: int = 0):
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)
        self.start_step = start_step
        self.step = start_step
        self.max_steps = max_steps

    def __iter__(self):
        return self

    def __next__(self):
        if self.step >= self.max_steps:
            raise StopIteration
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            batch = next(self.iter)
        self.step += 1
        return batch

    def __len__(self):
        return self.max_steps - self.start_step


# --------------------------------------------------------------------------- #
# Validation util
# --------------------------------------------------------------------------- #
def log_validation(vae, text_encoder, tokenizer, denoiser, args, accelerator, weight_dtype, step):
    accelerator.print(f"Running validation at step {step}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        denoiser=denoiser,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    images = []
    gen = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    for prompt in args.validation_prompts:
        ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(accelerator.device.type)
        with ctx:
            images.append(pipe(prompt, num_inference_steps=20, generator=gen).images[0])

    grid = make_image_grid(images, 1, len(images))
    grid.save(Path(args.output_dir) / f"val_grid_step_{step}.png")
    for tr in accelerator.trackers:
        if tr.name == "tensorboard":
            tr.writer.add_images("validation", np.stack([np.asarray(i) for i in images]), step, dataformats="NHWC")

    del pipe
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def train_ldm(args):
    # Seed / dirs
    if args.seed is not None:
        set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Accelerate setup
    proj_cfg = ProjectConfiguration(
        project_dir=args.output_dir,
        automatic_checkpoint_naming=True,
        total_limit=args.checkpoint_total_limit,
        iteration=0,
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=proj_cfg,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        split_batches=False,
        step_scheduler_with_optimizer=False,
    )
    accelerator.init_trackers(os.path.basename(args.output_dir))

    # Models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_clip)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_clip).eval()
    text_encoder.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae).eval()
    vae.requires_grad_(False)

    sd_path = "CompVis/stable-diffusion-v1-4"
    denoiser = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    if args.enable_xformers_memory_efficient_attention:
        denoiser.enable_xformers_memory_efficient_attention()
    denoiser.train()

    # EMA
    if args.use_ema:
        ema_denoiser = EMAModel(
            denoiser.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=denoiser.config,
            foreach=args.foreach_ema,
        )

    # Optimizer
    opt_cls = torch.optim.AdamW
    if args.use_8bit_adam:
        import bitsandbytes as bnb

        opt_cls = bnb.optim.AdamW8bit
    optimizer = opt_cls(
        denoiser.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # LR scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Data
    dataset = build_webdataset(args, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=False,  # handled by webdataset
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Wrap with Accelerator
    denoiser, optimizer, lr_scheduler = accelerator.prepare(denoiser, optimizer, lr_scheduler)

    # Precision-casting of frozen models
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    weight_dtype = dtype_map.get(accelerator.mixed_precision, "fp32")
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        ema_denoiser.to(accelerator.device)

    accelerator.print("Running training")
    accelerator.print(f"  Batch size = {args.train_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            start_step = 0
        else:
            # Restore denoiser weights, optimizer state_dict, scheduler state, random states
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            start_step = global_step
    else:
        start_step = 0

    dataloader_wrapper = DataloaderMaxSteps(dataloader, args.max_train_steps, start_step=start_step)
    for step, batch in (
        progress_bar := tqdm(
            enumerate(dataloader_wrapper, start=start_step),
            initial=start_step,
            total=args.max_train_steps,
            desc="Training",
        )
    ):
        with accelerator.accumulate(denoiser):
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(accelerator.device)

            with accelerator.autocast():
                latent_mean_logvar = vae._encode(pixel_values)
            
            # Ensure that the latents are sampled at float32
            posterior = DiagonalGaussianDistribution(latent_mean_logvar.to(torch.float32))
            latents = vae.config.scaling_factor * posterior.sample()

            noise = torch.randn_like(latents)
            if args.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += args.noise_offset * torch.randn(
                    (latents.size(0), latents.size(1), 1, 1), device=latents.device
                )
            if args.input_perturbation: # like data augmentation / regularization
                new_noise = noise + args.input_perturbation * torch.randn_like(noise)

            # random timestep per image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), 
                dtype = torch.int64,
                device=latents.device
            )
            # Add noise to the latents according to the noise magnitude at each timestep
            # forward diffusion process
            noisy_latents = noise_scheduler.add_noise(
                latents,
                new_noise if args.input_perturbation else noise,
                timesteps
            )
            
            with accelerator.autocast():
                # Get the text embedding for conditioning
                # torch.layer_norm returns fp32, so the output of text_encoder is fp32 rather than bf16

                enc_h = text_encoder(input_ids, return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.dream_training:
                    noisy_latents, target = compute_dream_and_update_latents(
                        denoiser,
                        noise_scheduler,
                        timesteps,
                        noise,
                        noisy_latents,
                        target,
                        enc_h,
                        args.dream_detail_preservation,
                    )

                pred = denoiser(noisy_latents, timesteps, enc_h, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    w = torch.minimum(snr, args.snr_gamma * torch.ones_like(snr))
                    if noise_scheduler.config.prediction_type == "epsilon":
                        w = w / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        w = w / (snr + 1)
                    loss = (F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3)) * w).mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            if args.use_ema:
                ema_denoiser.step(denoiser.parameters())
            global_step += 1
            accelerator.log({"train_loss": loss.item()}, step=global_step)

            # Checkpoint
            if global_step % args.checkpointing_steps == 0:
                ckpt = Path(args.output_dir) / f"checkpoint-{global_step}"
                accelerator.save_state(str(ckpt))
                accelerator.info(f"Checkpoint saved → {ckpt}")

            # Validation
            if (
                args.validation_prompts
                and global_step % args.validation_steps == 0
                and accelerator.is_local_main_process
            ):
                if args.use_ema:
                    ema_denoiser.store(denoiser.parameters())
                    ema_denoiser.copy_to(denoiser.parameters())
                log_validation(vae, text_encoder, tokenizer, denoiser, args, accelerator, weight_dtype, global_step)
                if args.use_ema:
                    ema_denoiser.restore(denoiser.parameters())

    # Final save ------------------------------------------------------------- #
    denoiser = accelerator.unwrap_model(denoiser)
    if args.use_ema:
        ema_denoiser.copy_to(denoiser.parameters())

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        vae=vae,
        denoiser=denoiser,
        revision=args.revision,
        variant=args.variant,
    )
    pipe.save_pretrained(args.output_dir)
    accelerator.success("Training complete ✅  Model saved.")


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    sys.exit(train_ldm(args))
