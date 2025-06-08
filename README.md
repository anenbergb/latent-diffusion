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


# Latent Diffusion Model Overview
