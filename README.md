# Latent Diffusion (LDM)
This repository provides a from-scratch implementation of latent diffusion model training with the goal of gaining a deeper understanding of diffusion architectures (e.g., U-Net, DiT) and scheduling methods. Both the model and diffusion process are custom-implemented and trained from the ground up. All experiments are constrained to a single NVIDIA RTX 5090 GPU (32 GB VRAM), so the training process is optimized for efficiency under limited compute. The objective is to achieve the highest possible model quality within this constraint.


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

# Diffusion Model Architectures

## Stable Diffusion - U-Net

## Diffusion Transformer (DiT)
- https://arxiv.org/abs/2212.09748
- https://github.com/facebookresearch/DiT
- https://huggingface.co/facebook/DiT-XL-2-256