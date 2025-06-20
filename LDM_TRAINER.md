# Latent Diffusion Model Trainer
The objective of the "from-scratch" LDM trainer is to maximize the quality of the resulting diffusion model while operating under the constraint of training on a single NVIDIA RTX 5090 GPU (32 GB VRAM) within a one-week time budget. These constraints will underly the choice of hyperparameters such as the batch size, the number of training iterations, the learning rate schedule, the image resolution, and model size.


Custom implementations of the diffusion model architecture (e.g., U-Net or DiT) and diffusion scheduler will be developed and trained from scratch. However, a pretrained VAE and CLIP text encoder will be used to reduce training complexity. For instance, the VAE may be sourced from the Stable Diffusion v1.4 model (`stabilityai/sd-vae-ft-mse`), and the CLIP encoder from the `laion/CLIP-ViT-L-14-laion2B-s32B-b82` checkpoint.


## LDM Trainer Concepts

### Gradient Accumulation
Gradient accumulation is a technique where you can train on bigger batch sizes than your machine would normally be able to fit into memory.
This is done by accumulating gradients over several batches, and only stepping the optimizer after a certain number of batches have been performed.

- https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation
- https://huggingface.co/docs/accelerate/en/concept_guides/gradient_synchronization
- https://huggingface.co/docs/accelerate/v1.8.0/en/package_reference/utilities#accelerate.utils.GradientAccumulationPlugin

### CLIP input caption standardization
CLIP can only process at most "model_max_length" tokens (77 tokens) so the tokenizer is forced to truncate long text prompts and pad short text prompts such that the token length is always 77.
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

### V-Objective
The v-objective trains the model to predict a combination of noise and the clean image, rather than just one or the other. It's mathematically equivalent to the other objectives under perfect conditions, but in practice, it improves stability and performance — making it a smart choice for models like Stable Diffusion v2.

V-Objective was introduced in in [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512).


Traditional diffusion models (DDPMs) are typically trained to predict:

**Noise objective**:  
Predict the added noise `ε` directly from a noisy image:

<img src="https://github.com/user-attachments/assets/71d92227-ba18-49f3-8370-cec1db2211b5" width="150"/>

**Denoised image objective**:  
Predict the original clean image `x₀` from the noisy input.

The **v-objective** reparameterizes the the denoising process and trains the model to predict:

<img src="https://github.com/user-attachments/assets/e27a60ac-e60f-47d3-81d9-59d3eb9ae1c2" width="200"/>

This is a linear combination of the noise ε and the clean image x₀, using the coefficients:
- $alpha_t = \sqrt{average(\alpha_t)}$
- $sigma_t = \sqrt{1-average(\alpha_t)}$

The training loss becomes:

<img src="https://github.com/user-attachments/assets/1cbfd9a6-7119-4854-b714-58a40d68019e" width="300"/>

####  Advantages of V-Objective
- Improved performance: Models trained with v-objective often generate higher quality samples.
- Better gradient behavior: It avoids some numerical instabilities of ε or x₀ prediction.
- Cleaner sampling: The v-objective aligns better with sampling methods like DDIM or PNDM.


## Inference Concepts
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

CFG was originally introduced in https://arxiv.org/abs/2207.12598.

## Evaluation
Table 2 from [LDM paper](https://arxiv.org/abs/2112.10752) reports the FID and Inception Scores for the 1.45B parameter LDM-KL-8 model given 250 DDIM Steps.
The performance is greatly improved by applying classifier-free diffusion guidance. `LDM-KL-8-G` refers to the model with classifier-free diffusion guidance.

| Method     | FID   | IS    | N_params |                      |
|------------|-------|-------|----------|----------------------|
| LDM-KL-8   | 23.31 | 20.03 | 1.45B    | 250 DDIM steps       |
| LDM-KL-8-G | 12.63 | 30.29 | 1.45B    | 250 DDIM steps, CFG  |


### FID
### Inception Score

## LDM Training Configuration
### Reference Model Configs

#### Latent Diffusion LDM-8 model
The original Latent Diffusion Model (LDM-8) from 

The tables in the Appendix of the original [LDM paper](https://arxiv.org/abs/2112.10752) show that a much higher batch size of 680 was required to train a conditional LDM-8 
for text-to-image generation on the LAION dataset as compared to the relatively small 96 batch size used to train an unconditional LDM-8 on the CelebA dataset.

<img src="https://github.com/user-attachments/assets/74740ad5-1883-4bfa-8d12-814088e86176" width="800"/>

Conditional LDMs require very large batch sizes (e.g., 680–1200) primarily due to:

1. Conditioning Signal Variability

- Text prompts or class labels introduce extra variance in the input space.
- The model needs to see many diverse conditionings per step to learn good alignments.
- Larger batches better capture that diversity in each gradient update.

2. Stability in Classifier-Free Guidance (CFG)

- Text-to-image models often use classifier-free guidance, which includes both:
  - Conditional (text-encoded) samples
  - Unconditional (empty prompt) samples
- Small batch sizes make the unconditional path gradients noisy, leading to unstable training.
- CFG benefits from large batches for stable mean gradients.

3. Better Signal-to-Noise Ratio in Latent Space

- Latent space training is more sensitive to noise.
- Small changes in the latent space can have outsized effects on the decoded image.
- Larger batch sizes stabilize training by reducing the variance of gradients.

4. Empirical Scaling Laws

- The LDM paper (and related works like Imagen, DALL·E 2) observed that:
  - Larger batch sizes improve sample quality
  - Help reduce overfitting
  - Enable faster convergence


The LDM-8 on CelebA was Trained with Batch Size 96 because:

- It's an unconditional generation task — no text or label conditioning.
- The input distribution is simpler (mostly faces).
- Smaller batch sizes suffice because:
  - No CFG is used
  - Less variance in input
  - Lower risk of overfitting
  - Training is more stable

Summary
- Conditional LDMs (especially text-to-image) benefit from large batch sizes ≥ 512
- Unconditional models can often train well with batch sizes 64–128

#### Stable Diffusion V1 - V1.5
SD V1.1 [CompVis/stable-diffusion-v1-1](https://huggingface.co/CompVis/stable-diffusion-v1-1)
- trained for 237k steps at resolution 256x256 on laion2B-en, followed by 194k steps at resolution 512x512 on laion-high-resolution (170M examples from LAION-5B with resolution >= 1024x1024)
- global batch size of 2048
- AdamW optimizer
- Learning rate: warmup to 1e-4 for 10k steps and then kept constant

SD V1.4 [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- initialized from SD V1.2
- fine-tuned for additional 225k steps at resolution 512x512 on "laion-aesthetics v2 5+" and 10% dropping of the text-conditioning to improve classifier-free guidance sampling.

#### Stable Diffusion V2
[stabilityai/stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2)


## Speeding up Runtime
- https://huggingface.co/docs/diffusers/en/tutorials/fast_diffusion
- https://pytorch.org/blog/accelerating-generative-ai-3/


## Observations from initial training runs
- a larger batch size stabilizes the gradients. Training on small batch size results in very noisy generated images.