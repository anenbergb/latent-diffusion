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


# Latent Diffusion Model Traine


### CLIP input caption standardization
CLIP can only process at most "model_max_length" tokens (77 tokens) so the tokenizer is forced to
truncate long text prompts and pad short text prompts such that the token length is always 77.
CLIP was likely trained with fixed 77 token sequences to maximize training speed, simplify architecture (fixed positional embeddings), and support large-batch contrastive learning — all without significantly sacrificing caption quality or alignment.

### Offset Noise

The [Crosslabs blog post](https://www.crosslabs.org//blog/diffusion-with-offset-noise) recommends
injecting a tiny random mean component "offset noise" of the form
```python
noise = torch.randn_like(latents) + 0.1 * torch.randn(latents.shape[0], latents.shape[1], 1, 1)
```
during training to break the hard zero-mean prior inherited from the standard diffusion schedule, giving the network the freedom to learn and reproduce full-range brightness.

The recommended offset noise range is `noise_offset: 0.05 to 0.2`

Recent work shows you can remove the same bias by redesigning the noise schedule (zero-terminal-SNR) or by using v-prediction.

### Input Perturbation

Acts as a regularizer / data-augmentation: makes the target ε-prediction task a bit harder, discourages over-fitting during fine-tune and improves robustness to small latent perturbations. 
```python
ε̂  = ε + ρ · 𝒩(0, I)              # ε̂ has variance 1+ρ²
noisy_latent = α_t * latent + σ_t * ε̂
target       = ε                  # we still ask the network to predict the *original* ε
```
The model now sees slightly “off-schedule” samples whose variance no longer matches the textbook β-schedule.
It must therefore become locally smoother (Lipschitz) in order to do well on the perturbed inputs, which empirically:
- reduces over-fitting on very small data (DreamBooth / LoRA)
- improves detail retention when guidance scale ≫ 1
- has negligible extra compute cost (one extra randn_like).

Input perturbation is not necessary when training with large datasets taht already sufficiently regularize the  model. It can be introduced if overfitting is observed. 
A suggested input perturbation noise range could be `0.05 to 0.15`

### Classifier-Free Guidance (CFG)
Classifier-Free Guidance is a technique used during inference (sampling) with diffusion models to improve prompt adherence. 

During inference, CFG modifies the denoising process to steer the model more toward the prompt.
It does this by running the denoising model twice:
1. Once with the actual text prompt (conditional)
2. Once with an empty prompt (unconditional)
It then interpolates between these two predictions to produce the final result:
```python
x_t_cond   = model(x_t, prompt)       # prediction given prompt
x_t_uncond = model(x_t, "")           # prediction given empty string

x_t_guided = x_t_uncond + w * (x_t_cond - x_t_uncond)
```
Where w is the guidance scale, often called CFG scale.
CFG scale
- 1-3: 	Weak guidance, more creative/stochastic outputs, prompt may be ignored
- 5-7.5: Balanced guidance; common default (e.g. 7.5 in Stable Diffusion)
- 10+: Strong guidance; improves adherence, but can cause artifacts, over-saturation, or loss of diversity
- 20+: Often unstable; results may collapse or contain noise

When CFG is high during inference, the model is being extrapolated outside the distribution it was trained on. In this case, it could be smart to train the model with v-prediction because v-prediction tends to be more numerically stable and better behaved under large extrapolation — it reduces artifacts, color drift, and instability.

### Epsilon-Prediction vs. V-Prediction Noise

The forward diffusion process at timestep (t) can be expressed as:
$$
x_t = alpha_t \cdot x_0 + sigma_t \cdot \varepsilon
$$

Where:
- $x_0$ is the original (clean) latent/image,
- $\varepsilon \sim \mathcal{N}(0, I)$ is standard Gaussian noise,
- $x_t$ is the noisy version at timestep \( t \),
- $alpha_t$ and $sigma_t$ are known scalars from the noise schedule.

#### Epsilon-Prediction

In this mode, the denoising model predicts the noise vector $\varepsilon$ that was added to the input during the forward diffusion process. The training objective minimizes the mean squared error (MSE) between the predicted noise and the actual noise.


- This is the original approach used in DDPM and Stable Diffusion v1.
- It places more weight on high-noise (late) timesteps, which can affect learning stability.

#### V-Prediction
V-prediction was introduced in [Google's Imagen paper](https://arxiv.org/abs/2205.11487).
Here, the model predicts a "velocity" vector, which is a linear combination of the noise and the original (clean) latent input.

$$v = alpha_t \cdot \varepsilon - sigma_t \cdot x_0$$

This can be rearranged to recover either $x_0$ or $\varepsilon$ during inference:

- To recover $x_0$:
$$x_0 = \frac{\alpha_t \cdot x_t - \sigma_t \cdot v}{\alpha_t^2 + \sigma_t^2}$$

- To recover $\varepsilon$:

$$\varepsilon = \frac{\sigma_t \cdot x_t + \alpha_t \cdot v}{\alpha_t^2 + \sigma_t^2}$$

#### Why Use V-Prediction?

- It balances the training loss across timesteps, improving numerical stability.
- It helps the model generalize better to low-noise or high-noise conditions.
- It performs better under high classifier-free guidance (CFG) during inference.

### Min-SNR
https://arxiv.org/abs/2303.09556
Compute loss-weights as per Section 3.4 of
Since we predict the noise instead of x_0, the original formulation is slightly changed.
This is discussed in Section 4.2 of the same paper.


### Gradient Accumulation
Gradient accumulation is a technique where you can train on bigger batch sizes than your machine would normally be able to fit into memory. This is done by accumulating gradients over several batches, and only stepping the optimizer after a certain number of batches have been performed.

https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation