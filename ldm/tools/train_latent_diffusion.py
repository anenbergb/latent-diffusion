import argparse
import os
from pathlib import Path
import sys
from typing import Optional
import inspect
import json
from PIL import Image
import shutil


import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import logging
from braceexpand import braceexpand
import webdataset as wds

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed, TorchDynamoPlugin
import diffusers
from diffusers import (
    AutoencoderKL,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.configuration_utils import ConfigMixin
from transformers import CLIPTextModel, CLIPTokenizer
from huggingface_hub import hf_hub_download

from ldm.tools.filter_dataset import CaptionFilter
from ldm.unet import UNet
from ldm.utils import get_cosine_schedule_with_warmup_then_constant


def parse_args():
    parser = argparse.ArgumentParser("Latent Diffusion Trainer")

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
        "--diffusion_model",
        type=str,
        default="UNet",
        choices=[
            "UNet",
        ],
        help="Name of the diffusion_model model to use. ",
    )
    parser.add_argument(
        "--hf_model_repo_id",
        type=str,
        help="Override the choice of diffusion_model to instead use the model "
        "from the HF repo ID e.g. 'CompVis/stable-diffusion-v1-4'.",
    )
    parser.add_argument(
        "--hf_model_subfolder",
        type=str,
        default="unet",
        help="HF repo subfolder for the diffusion_model. "
        "e.g. 'unet' for hf_model_repo_id: 'CompVis/stable-diffusion-v1-4' and "
        "'transformer' for hf_model_repo_id: 'facebook/DiT-XL-2-256'.",
    )
    parser.add_argument(
        "--diffusion_scheduler",
        type=str,
        default="ddpm",
        choices=[
            "ddpm",
        ],
        help="Diffusion scheduler to use. Currently only 'ddpm' is supported.",
    )
    parser.add_argument(
        "--hf_scheduler_repo_id",
        type=str,
        help="HF repo ID for the diffusion scheduler config. "
        "If not provided, the default scheduler config will be used based on the diffusion_scheduler argument.",
    )
    parser.add_argument(
        "--hf_scheduler_subfolder",
        type=str,
        default="scheduler",
        help="HF repo subfolder for the diffusion scheduler config. "
        "e.g. 'scheduler' for hf_scheduler_repo_id: 'CompVis/stable-diffusion-v1-4'.",
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
    parser.add_argument(
        "--caption_filters",
        type=str,
        nargs="+",
        required=False,
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
        default=[
            0.4,
        ],
        help="Cosine similarity thresholds (0.0-1.0) for each caption filter. "
        "Must provide one threshold per filter, or a single threshold to apply to all filters, "
        "or none to use default 0.5 for all. "
        "Higher values = stricter filtering (more similar to filter text required). "
        "Typical values: 0.3-0.7. Examples: --caption_filter_thresholds 0.4 0.6 0.5 (individual) "
        "or --caption_filter_thresholds 0.5 (same for all)",
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
    parser.add_argument(
        "--sentence_transformer_batch_size",
        type=int,
        default=1000,
        help="Number of captionst to process in a single batch",
    )

    parser.add_argument(
        "--sentence_transformer_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the SentenceTransformer model on. "
        "Choose 'cuda' for GPU acceleration (requires CUDA-compatible GPU) or 'cpu' for CPU execution. "
        "Default: 'cpu'",
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
        default="constant_with_warmup",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "cosine_with_warmup_then_constant",
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
    parser.add_argument(
        "--lr_constant_steps",
        type=int,
        required=True,
        help="Number of steps to hold the learning rate at max after warmup. "
        "Only relevant for cosine_with_warmup_then_constant scheduler.",
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
        "--enable_torch_compile", action="store_true", help="Enable torch.compile for the diffusion model."
    )
    parser.add_argument("--torch_dynamo_reset", action="store_true", help="Reset torch dynamo state before training.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=("epsilon", "v_prediction"),
        help="The prediction_type that shall be used for training.Choose between 'epsilon' or 'v_prediction'",
    )
    parser.add_argument(
        "--text_conditioning_dropout",
        type=float,
        default=0.1,
        help="Drop the text-conditioning embeddings with this probability (0.0 = no dropout).",
    )

    # --- misc / logging / checkpoints -------------------------------------- #
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save every N steps.")
    parser.add_argument(
        "--milestone_checkpoints", type=int, nargs="*", default=[], help="Save checkpoints at these steps."
    )
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
    parser.add_argument(
        "--generations_per_val_prompt",
        type=int,
        default=1,
        help="Number of images to generate per validation prompt.",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps to run for image generation.",
    )
    parser.add_argument(
        "--classifier_free_guidance_scale",
        type=int,
        default=7.5,
        help="Classifier-free guidance scale for inference (7.5 is a good default).",
    )
    parser.add_argument("--mixed_precision", choices=["fp32", "fp16", "bf16"], default="bf16", help="AMP dtype.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Worker processes for DataLoader. "
        "This should be less than or equal to the number of .tar files that WebDataset is loading from.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned", help="Checkpoint / TB dir.")

    args = parser.parse_args()

    # Validate arguments
    if args.caption_filters:
        if len(args.caption_filter_thresholds) == 1:
            # Single threshold provided - apply to all filters
            logging.info(
                f"Applying single threshold {args.caption_filter_thresholds[0]} to all {len(args.caption_filters)} filters"
            )
            args.caption_filter_thresholds = args.caption_filter_thresholds * len(args.caption_filters)
        elif len(args.caption_filters) != len(args.caption_filter_thresholds):
            parser.error(
                f"Number of caption filters ({len(args.caption_filters)}) must match "
                f"number of thresholds ({len(args.caption_filter_thresholds)}), "
                f"or provide exactly one threshold to apply to all filters"
            )
    return args


def build_diffusers_model_registry():
    registry = {}
    for name, obj in inspect.getmembers(diffusers.models, inspect.isclass):
        # Ensure it's a diffusers model and a subclass of ConfigMixin
        if issubclass(obj, ConfigMixin) and obj.__module__.startswith("diffusers.models"):
            registry[name] = obj
    return registry


def load_hf_model_from_config(repo_id: str, subfolder: Optional[str] = None, config_name: str = "config.json"):
    config_path = hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=config_name)
    with open(config_path) as f:
        config = json.load(f)

    class_name = config["_class_name"]
    registry = build_diffusers_model_registry()

    if class_name not in registry:
        raise ValueError(f"Unknown model class: {class_name}")

    model_class = registry[class_name]
    return model_class.from_config(config)


def build_diffusers_scheduler_registry():
    registry = {}
    for name, obj in inspect.getmembers(diffusers.schedulers, inspect.isclass):
        if issubclass(obj, ConfigMixin) and obj.__module__.startswith("diffusers.schedulers"):
            registry[name] = obj
    return registry


def load_hf_scheduler_from_config(
    repo_id: str, subfolder: str = "scheduler", config_name: str = "scheduler_config.json"
):
    config_path = hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=config_name)
    with open(config_path) as f:
        config = json.load(f)

    class_name = config["_class_name"]
    registry = build_diffusers_scheduler_registry()

    if class_name not in registry:
        raise ValueError(f"Unknown scheduler class: {class_name}")

    scheduler_class = registry[class_name]
    return scheduler_class.from_config(config)


def build_webdataset(args, tokenizer):
    # Expand brace patterns into concrete *.tar paths
    tar_files = []
    for spec in args.dataset_tar_specs:
        tar_files.extend(braceexpand(spec))
    if not tar_files:
        raise FileNotFoundError("No tar files resolved from --dataset_tar_specs")

    logging.info(f"Found {len(tar_files)} tar files to load for training.")

    # WebDataset decode logic
    # https://github.com/webdataset/webdataset/blob/main/webdataset/autodecode.py#L299
    dataset = wds.WebDataset(
        tar_files,
        repeat=False,
        shardshuffle=len(tar_files),
        detshuffle=True,
        seed=args.seed,
    ).shuffle(args.shuffle_buffer)

    if args.caption_filters:
        caption_filter = CaptionFilter(
            filter_texts=args.caption_filters,
            filter_thresholds=args.caption_filter_thresholds,
            model_name=args.sentence_transformer_model_name,
            device=args.sentence_transformer_device,
        )

        def batched_filter_fn(batch):
            """each sample in the batch is an undecoded dict"""
            captions = [json.loads(sample["json"].decode("utf-8")).get("caption", "") for sample in batch]
            keep_flags = caption_filter.filter_batch(captions)
            return [sample for sample, passed in zip(batch, keep_flags) if passed]

        dataset = dataset.listed(args.sentence_transformer_batch_size).map(batched_filter_fn).unlisted()

    dataset = dataset.decode("pilrgb").to_tuple("jpg", "json")

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


def get_vae_downscale(config: dict) -> int:
    """Return the spatial reduction factor of a Diffusers AutoencoderKL."""
    n_blocks = len(config["down_block_types"])
    return 2 ** (n_blocks - 1)  # last block keeps resolution


@torch.inference_mode()
def generate(
    tokenizer,
    text_encoder,
    vae,
    diffusion_model,
    noise_scheduler,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 256,
    width: int = 256,
    seed: int | None = None,
    device: torch.device = torch.device("cuda"),
    weight_dtype: torch.dtype = torch.bfloat16,
    return_type: str = "pil",
    is_hf_diffusion_model: bool = False,
) -> Image.Image:
    # Prompt -> embeddings (classifier-free guidance = cond + uncond)
    text_inputs = tokenizer(
        [prompt], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to(device=device)

    uncond_inputs = tokenizer(
        [""], max_length=tokenizer.model_max_length, padding="max_length", return_tensors="pt"
    ).input_ids.to(device=device)

    text_embeds = text_encoder(text_inputs, return_dict=False)[0].to(dtype=weight_dtype)
    uncond_embeds = text_encoder(uncond_inputs, return_dict=False)[0].to(dtype=weight_dtype)
    embeds = torch.cat([uncond_embeds, text_embeds], dim=0)  # (2, T, 768)

    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    # Latent tensor initialized with pure Gaussian noise
    gen = None if seed is None else torch.Generator(device=device).manual_seed(seed)

    latent_channels = vae.config.latent_channels
    latent_downscale = get_vae_downscale(vae.config)  # 8
    latents = torch.randn(
        (1, latent_channels, height // latent_downscale, width // latent_downscale),
        generator=gen,
        device=device,
        dtype=weight_dtype,
    )  # (1,4,32,32)
    latents *= noise_scheduler.init_noise_sigma  # match training scale

    # Diffusion (reverse) loop
    for timestep in noise_scheduler.timesteps:
        # Duplicate for (uncond, cond) batches
        latent_in = torch.cat([latents] * 2, dim=0)  # (2,4,32,32)
        latent_in = noise_scheduler.scale_model_input(latent_in, timestep)

        if is_hf_diffusion_model:
            noise_pred = diffusion_model(latent_in, timestep, embeds, return_dict=False)[0]
        else:
            noise_pred = diffusion_model(latent_in, timestep, embeds)

        eps_uncond, eps_cond = noise_pred.chunk(2)

        # Classifier-free guidance
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        latents = noise_scheduler.step(eps, timestep, latents).prev_sample

    # Latents -> image space through VAE decoder
    latents = latents / vae.config.scaling_factor  # SD latent scaling
    image = vae.decode(latents, return_dict=False)[0]

    # Post-process [-1,1] -> [0,1] -> [0,255]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (255 * image.to(device="cpu")).to(torch.uint8).squeeze(0)
    if return_type == "pil":
        return Image.fromarray(image.permute(1, 2, 0).numpy())
    return image


def log_validation(
    tokenizer,
    text_encoder,
    vae,
    diffusion_model,
    noise_scheduler,
    args,
    accelerator,
    train_step,
    is_hf_diffusion_model: bool = False,
):
    accelerator.print(f"Running validation at step {train_step}...")
    diffusion_model.eval()
    images = []
    for prompt in args.validation_prompts:
        with accelerator.autocast():
            for i in range(args.generations_per_val_prompt):
                image = generate(
                    tokenizer,
                    text_encoder,
                    vae,
                    diffusion_model,
                    noise_scheduler,
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.classifier_free_guidance_scale,
                    height=args.resolution,
                    width=args.resolution,
                    seed=args.seed + i if args.seed is not None else None,
                    device=accelerator.device,
                    return_type="torch",
                    is_hf_diffusion_model=is_hf_diffusion_model,
                )
                images.append(image)

    nrow = args.generations_per_val_prompt if args.generations_per_val_prompt > 1 else 3
    image_grid = make_grid(images, nrow=nrow, padding=0, value_range=(0, 255))  # (C,H,W)
    image_grid_pil = Image.fromarray(image_grid.permute(1, 2, 0).numpy())
    image_grid_pil.save(Path(args.output_dir) / f"val_images_step_{train_step}.png")
    tracker = accelerator.get_tracker("tensorboard")
    tracker.log_images({"validation": image_grid}, train_step, dataformats="CHW")
    torch.cuda.empty_cache()
    diffusion_model.train()


def log_validation_prompts(
    accelerator: Accelerator,
    validation_prompts: Optional[list[str]] = None,
):
    if validation_prompts is None:
        return

    for i, prompt in enumerate(validation_prompts):
        accelerator.log({f"val_prompt_{i}": prompt})


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
        iteration=0,  # the current save iteration
    )
    dynamo_plugin = TorchDynamoPlugin(
        backend="inductor",
        mode="max-autotune",
        fullgraph=False,
        use_regional_compilation=True,
    )
    # https://huggingface.co/docs/diffusers/en/tutorials/fast_diffusion#torchcompile
    if args.enable_torch_compile:
        print("Enabling torch.compile for the diffusion model...")
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        # torch._inductor.config.triton.cudagraphs = False
        if args.torch_dynamo_reset:
            print("Resetting torch dynamo state...")
            torch._dynamo.reset()

    # https://huggingface.co/docs/accelerate/en/concept_guides/gradient_synchronization
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=proj_cfg,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        split_batches=False,
        step_scheduler_with_optimizer=False,
        dynamo_plugin=dynamo_plugin if args.enable_torch_compile else None,
    )
    accelerator.init_trackers(os.path.basename(args.output_dir))

    # Models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_clip)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_clip).eval()
    text_encoder.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae).eval()
    vae.requires_grad_(False)

    vae.to(memory_format=torch.channels_last)
    vae = torch.compile(vae, backend="inductor", mode="max-autotune", fullgraph=True)

    is_hf_diffusion_model = bool(args.hf_model_repo_id)
    if args.hf_model_repo_id:
        diffusion_model = load_hf_model_from_config(
            repo_id=args.hf_model_repo_id,
            subfolder=args.hf_model_subfolder,
        )
    else:
        if args.diffusion_model == "UNet":
            diffusion_model = UNet(in_channels=vae.config.latent_channels)
        else:
            raise ValueError(f"Unknown diffusion model: {args.diffusion_model}")
    diffusion_model.to(memory_format=torch.channels_last)

    if args.hf_scheduler_repo_id:
        noise_scheduler = load_hf_scheduler_from_config(
            repo_id=args.hf_scheduler_repo_id,
            subfolder=args.hf_scheduler_subfolder,
        )
    else:
        raise NotImplementedError("Loading custom diffusion schedulers is not implemented yet.")

    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    if is_hf_diffusion_model and args.enable_xformers_memory_efficient_attention:
        assert args.enable_torch_compile is False, "xformers and torch.compile are not compatible yet."
        diffusion_model.enable_xformers_memory_efficient_attention()
    diffusion_model.train()

    # EMA
    if args.use_ema:
        ema_diffusion_model = EMAModel(
            diffusion_model.parameters(),
            model_cls=type(diffusion_model),
            model_config=diffusion_model.config,
            foreach=args.foreach_ema,
        )

    # Optimizer
    opt_cls = torch.optim.AdamW
    if args.use_8bit_adam:
        import bitsandbytes as bnb

        opt_cls = bnb.optim.AdamW8bit
    optimizer = opt_cls(
        diffusion_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # LR scheduler
    if args.lr_scheduler == "cosine_with_warmup_then_constant":
        lr_scheduler = get_cosine_schedule_with_warmup_then_constant(
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
            num_constant_steps=args.lr_constant_steps,
            last_epoch=-1,
        )
    else:
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
    diffusion_model, optimizer, lr_scheduler = accelerator.prepare(diffusion_model, optimizer, lr_scheduler)

    # Precision-casting of frozen models
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    weight_dtype = dtype_map.get(accelerator.mixed_precision, "fp32")
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        ema_diffusion_model.to(accelerator.device)

    accelerator.print("Running training")
    accelerator.print(f"  Batch size = {args.train_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")

    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if args.resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                dirs = os.listdir(checkpoint_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
                path = os.path.join(checkpoint_dir, dirs[-1]) if len(dirs) > 0 else None
            else:
                path = None
        # path should be formatated as /path/to/output_dir/checkpoints/checkpoint_XXXXX
        if path is None or not os.path.exists(path):
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            start_step = 0
        else:
            # Restore diffusion_model weights, optimizer state_dict, scheduler state, random states
            accelerator.print(f"Loading weights from checkpoint {path}")
            accelerator.load_state(path)
            start_step = int(path.split("_")[-1])
            accelerator.print(f"Resuming training from step {start_step}")
    else:
        start_step = 0

    log_validation_prompts(accelerator, args.validation_prompts)

    dataloader_wrapper = DataloaderMaxSteps(
        dataloader,
        args.max_train_steps * args.gradient_accumulation_steps,
        start_step=start_step * args.gradient_accumulation_steps,
    )

    # Prepare unconditional (null) embedding once to support text-conditioning dropout
    null_text_input_ids = tokenizer(
        [""], max_length=tokenizer.model_max_length, padding="max_length", return_tensors="pt"
    ).input_ids.to(accelerator.device)
    null_text_emb = text_encoder(null_text_input_ids, return_dict=False)[0].to(weight_dtype)

    # If args.gradient_accumulation_steps = 2, then 'step' is incremented every 2 micro-batches
    progress_bar = tqdm(range(0, args.max_train_steps), initial=start_step, desc="Training")
    step = start_step
    train_loss = 0.0
    for batch in dataloader_wrapper:
        with accelerator.accumulate(diffusion_model):  # accumulate gradients diffusion_model.grad
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
            input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)

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
            if args.input_perturbation:  # like data augmentation / regularization
                new_noise = noise + args.input_perturbation * torch.randn_like(noise)

            # random timestep per image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.size(0),),
                dtype=torch.int64,
                device=latents.device,
            )
            # Add noise to the latents according to the noise magnitude at each timestep
            # forward diffusion process
            noisy_latents = noise_scheduler.add_noise(
                latents, new_noise if args.input_perturbation else noise, timesteps
            )

            with accelerator.autocast():
                # Get the text embedding for conditioning
                # torch.layer_norm returns fp32, so the output of text_encoder is fp32 rather than bf16
                # https://docs.pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float32

                enc_h = text_encoder(input_ids, return_dict=False)[0].to(weight_dtype)

                # randomly drop text conditioning
                if args.text_conditioning_dropout > 0.0:
                    # Create a mask for dropping text conditioning
                    drop_mask = torch.rand(input_ids.shape[0], device=input_ids.device) < args.text_conditioning_dropout
                    if drop_mask.any():
                        # Replace the dropped text embeddings with null embedding
                        enc_h[drop_mask] = null_text_emb

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.dream_training:
                    noisy_latents, target = compute_dream_and_update_latents(
                        diffusion_model,
                        noise_scheduler,
                        timesteps,
                        noise.to(weight_dtype),
                        noisy_latents.to(weight_dtype),
                        target.to(weight_dtype),
                        enc_h,
                        args.dream_detail_preservation,
                    )

                # the final GroupNorm converts the activations to fp32
                if is_hf_diffusion_model:
                    pred = diffusion_model(noisy_latents.to(weight_dtype), timesteps, enc_h, return_dict=False)[0]
                else:
                    pred = diffusion_model(noisy_latents.to(weight_dtype), timesteps, enc_h)

            if args.snr_gamma is None:
                loss = F.mse_loss(pred.to(torch.float32), target.to(torch.float32), reduction="mean")
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                w = torch.minimum(snr, args.snr_gamma * torch.ones_like(snr))
                if noise_scheduler.config.prediction_type == "epsilon":
                    w = w / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    w = w / (snr + 1)
                loss = (F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3)) * w).mean()

            train_loss += loss.detach().item() / args.gradient_accumulation_steps
            accelerator.backward(loss)

        # end of accelerator.accumulate context
        if accelerator.sync_gradients:  # True only on the final micro-batch
            accelerator.clip_grad_norm_(diffusion_model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            current_lr = lr_scheduler.get_last_lr()[0]
            logs = {
                "loss/train": train_loss,
                "lr": current_lr,
            }
            step += 1
            accelerator.log(logs, step=step)
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            train_loss = 0.0

            if args.use_ema:
                ema_diffusion_model.step(diffusion_model.parameters())

            if step > 1 and (step % args.checkpointing_steps == 0 or step == args.max_train_steps):
                accelerator.project_configuration.iteration = step
                accelerator.save_state()
                # save EMA weights too
                # torch.save(ema_diffusion_model.state_dict(), ...)
                accelerator.print(f"Checkpoint {step} saved")

            if step in args.milestone_checkpoints:
                accelerator.project_configuration.automatic_checkpoint_naming = False
                accelerator.project_configuration.iteration = step
                accelerator.save_state(
                    output_dir=os.path.join(args.output_dir, "milestone_checkpoints", f"checkpoint_{step}")
                )
                accelerator.project_configuration.automatic_checkpoint_naming = True
                accelerator.print(f"Milestone Checkpoint {step} saved")

            # Validation
            if step > 1 and args.validation_prompts and step % args.validation_steps == 0:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_diffusion_model.store(diffusion_model.parameters())
                    ema_diffusion_model.copy_to(diffusion_model.parameters())
                log_validation(
                    tokenizer,
                    text_encoder,
                    vae,
                    diffusion_model,
                    noise_scheduler,
                    args,
                    accelerator,
                    step,
                    is_hf_diffusion_model=is_hf_diffusion_model,
                )
                if args.use_ema:
                    # Switch back to the original UNet parameters
                    ema_diffusion_model.restore(diffusion_model.parameters())

    # end of training loop
    if args.use_ema:
        diffusion_model = accelerator.unwrap_model(diffusion_model)
        ema_diffusion_model.copy_to(diffusion_model.parameters())

    accelerator.end_training()
    return 0


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    sys.exit(train_ldm(args))
