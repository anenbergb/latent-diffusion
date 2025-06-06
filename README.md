# latent-diffusion
From-scratch implementation of latent diffusion (stable diffusion)



## Installation
```
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install triton==3.3.1
pip install --pre xformers --index-url https://download.pytorch.org/whl/nightly/cu128


pip install -v --no-build-isolation git+https://github.com/facebookresearch/xformers.git@main#egg=xformers


<!-- pip install -v --no-build-isolation git+https://github.com/triton-lang/triton.git -->
```
### Run Formatting
```
ruff format llm
```

### Huggingface datasets

Set an alternate location for huggingface home
```
conda env config vars set HF_HOME=/media/bryan/ssd01/huggingface
conda env config vars list
```

## References
- https://github.com/CompVis/latent-diffusion
- https://github.com/Stability-AI/stablediffusion
- https://laion.ai/blog/large-openclip/ (OpenCLIP)
- https://github.com/CompVis/stable-diffusion
- https://github.com/huggingface/diffusers
- https://github.com/facebookresearch/xformers
- https://github.com/CompVis/taming-transformers


# Implementation plan

## Datasets

Download Open Images V7 from FiftyOne
https://docs.voxel51.com/dataset_zoo/datasets.html#open-images-v7


Training plan: 
- Train VAE on subset of OpenImages to prove that implementation works
- Scale VAE training to full OpenImages dataset
- Train LDM on ImageNet-1k subset
- Scale LDM training to full ImageNet-1k dataset

Afer class-conditional LDM training, try text-to-image training with LAION-400m dataset
- https://huggingface.co/datasets/laion/laion400m

## Models

## VAE / auto-encoder (frozen at diffusion time)
- Convolutional ResNet-style encoder/decoder exactly as in LDM (x8 spatial down-/up-sampling, 4 latent channels).

Pre-activation ResBlocks + GroupNorm → faster convergence
- SwiGLU activation in every conv/MLP (a no-brainer drop-in for SiLU/ReLU)
- Perceptual (LPIPS) + L2 reconstruction loss
- Exponential-moving-average (EMA) weights for the final checkpoint



Images are encoded through an encoder, which turns images into latent representations. The autoencoder uses a relative downsampling factor of 8 and maps images of shape H x W x 3 to latents of shape H/f x W/f x 4
Text prompts are encoded through the OpenCLIP-ViT/H text-encoder.
The output of the text encoder is fed into the UNet backbone of the latent diffusion model via cross-attention.
The loss is a reconstruction objective between the noise that was added to the latent and the prediction made by the UNet. We also use the so-called v-objective, see https://arxiv.org/abs/2202.00512.

Training Procedure Stable Diffusion v2 is a latent diffusion model which combines an autoencoder with a diffusion model that is trained in the latent space of the autoencoder. During training,

- Images are encoded through an encoder, which turns images into latent representations. The autoencoder uses a relative downsampling factor of 8 and maps images of shape H x W x 3 to latents of shape H/f x W/f x 4
- Text prompts are encoded through the OpenCLIP-ViT/H text-encoder.
- The output of the text encoder is fed into the UNet backbone of the latent diffusion model via cross-attention.
- The loss is a reconstruction objective between the noise that was added to the latent and the prediction made by the UNet. We also use the so-called v-objective, see https://arxiv.org/abs/2202.00512.

### Fine-Tuning Only the VAE Decoder

Fine-tuning only the decoder of a VAE allows for improved image reconstruction quality—such as sharper details and reduced artifacts—without modifying the latent space produced by the encoder. This ensures compatibility with existing diffusion models trained on the original encoder output. It is an efficient approach for enhancing output fidelity or adapting to new domains, and was used in official Stable Diffusion VAE fine-tuning (e.g., `sd-vae-ft-mse`).

### LPIPS Loss

LPIPS (Learned Perceptual Image Patch Similarity) is a perceptual loss that compares images based on deep feature representations extracted from a pre-trained network, rather than raw pixels. It better aligns with human perception and encourages sharper, more realistic reconstructions. LPIPS is commonly used in training VAEs, GANs, and diffusion autoencoders to improve visual fidelity.


## Hyperparameter settings in KL F8 VAE
Exponential Moving Average
- The Stable Diffusion VAE (KL-f8) was trained using EMA with a decay rate of `0.9999`.
- EMA helps stabilize training and improves reconstruction quality by smoothing recent parameter updates. It acts like a form of implicit ensemble smoothing.
- The publicly released VAE weights (e.g., `sd-vae-ft-mse`, `sdxl-vae`) are EMA-smoothed checkpoints, and all inference uses the EMA weights

Batch Size
- Stable Diffusion VAE (KL-f8) was trained with a batch size of 192, distributed across 16 NVIDIA A100 GPUs, with each GPU handling a batch of 12 images

8-bit Adam Optimizer
- 8-bit Adam provides significant memory and training speed improvements with minimal to no performance degradation when used for training VAEs, diffusion models, and transformers. Empirical studies show it matches 32-bit Adam on tasks like image synthesis and text generation. Minor stability issues can arise with very small batches or unusual training setups, but these are rare in typical diffusion pipelines.

