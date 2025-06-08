# Variational Autoencoder

<img src="https://github.com/user-attachments/assets/ad5eadf5-15cd-4c62-a991-a109066618a9" width="500"/>

Latent Diffusion Models (LDMs) use an autoencoder to compress high-dimensional images into a lower-dimensional latent space before applying the diffusion process.
This design introduces several key advantages:

- **Computational Efficiency**: Operating directly on pixel-space images (e.g., 256×256×3) is computationally expensive. By using an autoencoder with a downsampling factor (e.g., `f = 8`),
the model instead learns to denoise in a much smaller latent space (e.g., 32×32×4), drastically reducing memory and compute requirements.

- **Faster Training and Sampling**: With fewer spatial dimensions, the UNet used in the diffusion model processes fewer total elements, allowing faster iterations and
enabling high-resolution synthesis (e.g., 512×512 and above) on standard hardware.

- **Semantic Compression**: The encoder learns a compact representation that captures high-level semantic features (e.g., object structure, textures) while discarding
imperceptible pixel-level noise. This leads to more stable and semantically meaningful diffusion trajectories.

- **Modularity**: The autoencoder can be trained independently from the diffusion model, allowing for flexible reuse. Once trained, the encoder/decoder is frozen and
reused across tasks like class-conditional, text-to-image, and inpainting.

The autoencoder is a critical component in LDMs that enables efficient, scalable, and high-quality image synthesis by shifting the generative modeling process from 
pixel space to a learned latent space.

## KL F-8 VAE

In this project we will focus on the **KL F-8 VAE** because while VQ-GAN was effective for early latent diffusion, 
KL-VAEs became the preferred formulation for flexibility, quality, and compatibility with downstream tasks.

All Latent Diffusion Models (LDMs) trained and evaluated in the original [Latent Diffusion Models paper](https://arxiv.org/abs/2112.10752) 
used a VQ-GAN as the first-stage autoencoder. All results in the paper (e.g., ImageNet, LSUN, CelebA-HQ) rely on VQ-GAN latents.

### Comparison: KL VAE vs VQ-GAN

A **KL VAE** is a variational autoencoder that learns a continuous latent space regularized by KL divergence to a Gaussian prior, 
whereas a **VQ-GAN  (Vector-Quantized Generative Adversarial Network)** encodes inputs into a discrete latent space 
(grid of codebook indices) using vector quantization and enhances reconstruction quality with adversarial (GAN) training. 

In the [Latent Diffusion paper](https://arxiv.org/abs/2112.10752), VQ-GAN was used as the first-stage autoencoder because:

- It provided a **discrete, compressed latent space**, enabling efficient diffusion modeling.
- The **adversarial training** improved the sharpness and realism of reconstructions.
- It was compatible with **transformer-style architectures** and token-based modeling.

However, in later models like **Stable Diffusion v1/v2/SDXL**, VQ-GAN was replaced by a **KL-regularized VAE** (continuous latent space) because:
- **Continuous latents are fully differentiable**, enabling gradient-based finetuning (e.g., LoRA, DreamBooth).
- **No codebook quantization** means fewer artifacts (e.g., checkerboard patterns).
- **Simpler training and implementation**, without needing a codebook or straight-through estimators.
- Empirically, KL-VAEs achieved **comparable or better reconstruction metrics** (e.g., lower R-FID in Table 8 of the LDM paper).


| Aspect            | **KL VAE**                        | **VQ-GAN**                              |
|-------------------|-----------------------------------|------------------------------------------|
| Latent space      | **Continuous** (Gaussian)         | **Discrete** (quantized via codebook)   |
| Sampling          | Latents drawn from `N(0, I)`      | Latents selected from fixed codebook    |
| Differentiable    | ✅ Fully differentiable            | ❌ Non-differentiable (uses straight-through estimator) |
| Reconstruction    | Smooth, continuous, adaptable     | Crisper but may show tiling artifacts   |
| Finetuning (e.g. LoRA, DreamBooth) | ✅ Easy (gradients flow through latents) | ❌ Challenging due to quantization barrier |

### Evidence from Table 8 (Appendix F.1 of the LDM Paper)

Furthermore, the KL-VAE demonstrated better reconstruction metrics than the VQ-GAN at the same compression level. 
According to Table 8 in the LDM paper, KL-f8 achieved a lower R-FID and higher PSNR and SSIM, 
making it one of the best-performing and most efficient autoencoders evaluated.

| Type     | Downsampling Factor (f) | Latent Shape (z-shape) | R-FID ↓ | PSNR ↑ | SSIM ↑ |
|----------|--------------------------|-------------------------|---------|--------|--------|
| **KL**   | 8                        | 32×32×4                 | **0.87** | 24.08  | 0.68   |
| VQ-GAN   | 8                        | 32×32×4                 | 1.14    | 23.07  | 0.65   |
| **KL**   | 4                        | 64×64×4                 | 0.65    | **26.11** | **0.78** |
| VQ-GAN   | 4                        | 64×64×4                 | **0.58** | 25.46  | 0.75   |

- Lower **R-FID** indicates better perceptual similarity to real images.
- Higher **PSNR** and **SSIM** indicate more accurate and structurally faithful reconstructions.
- At `f = 8`, the **KL autoencoder outperforms VQ-GAN** in all three metrics.

### KL-VAE Architecture

- **Input resolution**: 256 × 256 RGB image  
- **Downsampling factor (f)**: 8  
  - Output latent shape: **32 × 32 × 4**  
- **Latent channels (c)**: 4  
- **Encoder**: ResNet-style CNN with spatial downsampling  
- **Decoder**: Symmetric upsampling CNN with skip connections  
- **Attention layers**: Optional self-attention at intermediate resolutions (e.g., 64×64)

### Network Architecture
```
Input (B, 3, 256, 256)
│
▼
Conv2d 3x3 (3 → 128) (B,128,256,256)
│
▼
╔══════════════════════════════╗
║ DownEncoderBlock2D × 4       ║
║ ┌──────────────────────────┐ ║
║ │ ResnetBlock2D × 2        │ ║
║ │ (with GroupNorm, SiLU)   │ ║
║ └──────────────────────────┘ ║
║ ┌──────────────────────────┐ ║
║ │ Downsample2D (stride=2)  │ ║
║ └──────────────────────────┘ ║
╚══════════════════════════════╝
│ [ -> (B,128,128,128) -> (B,256,64,64) -> (B,512,32,32) -> (B,512,32,32) ]
| 
▼
╔══════════════════════════════╗
║ UNetMidBlock2D               ║
║ ┌──────────────────────────┐ ║
║ │ ResnetBlock2D            │ ║
║ │ Self-Attention           │ ║
║ │ ResnetBlock2D            │ ║
║ └──────────────────────────┘ ║
╚══════════════════════════════╝
│ (B,512,32,32)
▼
GroupNorm → SiLU → Conv2d(512 → 8)
│ (B,8,32,32)
▼
QuantConv 1×1 (8 → 8)
│ (B,8,32,32)
▼
Sample from Diagonal Gaussian
│ (B,4,32,32)
▼
PostQuantConv 1×1 (4 → 4)
│ (B,4,32,32)
▼
Conv2d 3x3 (4 → 512)
│ (B,512,32,32)
▼
╔══════════════════════════════╗
║ UNetMidBlock2D               ║
║ ┌──────────────────────────┐ ║
║ │ ResnetBlock2D            │ ║
║ │ Self-Attention           │ ║
║ │ ResnetBlock2D            │ ║
║ └──────────────────────────┘ ║
╚══════════════════════════════╝
│ (B,512,32,32)
▼
╔══════════════════════════════╗
║ UpDecoderBlock2D × 4         ║
║ ┌──────────────────────────┐ ║
║ │ ResnetBlock2D × 3        │ ║
║ │ (with GroupNorm, SiLU)   │ ║
║ └──────────────────────────┘ ║
║ ┌──────────────────────────┐ ║
║ │Upsample2D (Conv + Interp)│ ║
║ └──────────────────────────┘ ║
╚══════════════════════════════╝
│ [ -> (B,512,64,64) -> (B,512,128,128) -> (B,256,256,256) -> (B,128,256,256) ]
▼
GroupNorm → SiLU → Conv2d(128 → 3)
│
▼
Reconstructed Image (B, 3, 256, 256)
```

### Sampling Latent Tensor
Given an input image of shape `(N,3,256,256)` the VAE encoder predicts mean and logvar tensors of shape `(N,4,32,32)` that define a Diagonal Gaussian Distribution. The standard deviation (std) can be computed from the logvar by `std = torch.exp(0.5 * logvar)`.
A latent embedding tensor of shape `(N,4,32,32)` can be sampled from this gaussian distribution with the following code
```python
sample = torch.randn_like(self.mean)
x = mean + std * sample
```

### KL-VAE Training Losses

The KL-F8 VAE is trained using a combination of losses:

#### KL Divergence Loss
- Encourages the latent distribution to follow a unit Gaussian (`N(0, I)`), enabling smooth and continuous sampling.  
- Helps regularize the latent space and stabilizes training.
- The latent distribution is diagonal, so there is no covariance between dimensions.

The VAE encoder predicts mean and logvar of shape `(N,4,32,32)` from which you can define
```python
std = torch.exp(0.5 * logvar)
var = torch.exp(logvar)
```

The KL divergence between two diagonal Gaussian distributions
- $q(z) = \mathcal{N}(\mu_1, \sigma_1^2)$  
- $p(z) = \mathcal{N}(\mu_2, \sigma_2^2)$

$$ D_{KL}(q \parallel p) = \frac{1}{2} \sum_i \left( \frac{(\mu_1 - \mu_2)^2}{\sigma_2^2} + \frac{\sigma_1^2}{\sigma_2^2} - 1 + \log \left( \frac{\sigma_2^2}{\sigma_1^2} \right) \right) $$

and in the special case of $p(z) = \mathcal{N}(0, 1)$

$$
D_{KL}\left( \mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1) \right) =
\frac{1}{2} \sum_i \left( \mu_i^2 + \sigma_i^2 - 1 - \log \sigma_i^2 \right)
$$

which can be expressed in python as
```python
kl_loss = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2, 3])
```
scale_factor: 1e-6

#### LPIPS Loss (Learned Perceptual Image Patch Similarity)
  Measures perceptual similarity using deep features extracted from a pretrained network (e.g., VGG).  
  Encourages reconstructions that *look* closer to the input from a human visual perspective, even if pixel values differ.

We use `lpips` package https://github.com/richzhang/PerceptualSimilarity
```python
perceptual_loss = lpips.LPIPS(net="vgg").eval()
p_loss = perceptual_loss(reconstructions, targets)
```
scale_factor: 0.5

#### L2 (Reconstruction) Loss
  Penalizes pixel-wise differences between the input and reconstruction.  
  Ensures structural accuracy but can lead to blurry outputs if used alone.
```python
rec_loss = F.mse_loss(reconstructions, targets)
```

#### Adversarial Loss (PatchGAN Discriminator)
- A discriminator network classifies image patches as real or fake.
- This loss sharpens outputs by forcing the decoder to generate more realistic textures and fine details.
- Discriminator loss isn't added to training until 50k iterations.

The [taming-transformers](https://github.com/CompVis/taming-transformers) `NLayerDiscriminator` (PatchGAN as in Pix2Pix) is used.
```
(N,3,256,256)
|
▼
Conv2d 4x4 stride=2 (3 → 64), LeakyReLU
| (N,64,128,128)
▼
Conv2d 4x4 stride=2 (64 → 128), BatchNorm2d, LeakyReLU
| (N,128,64,64)
▼
Conv2d 4x4 stride=2 (128 → 256), BatchNorm2d, LeakyReLU
| (N,256,32,32)
▼
Conv2d 4x4 stride=1 (256 → 512), BatchNorm2d, LeakyReLU
| (N,512,31,31)
▼
Conv2d 4x4 stride=1 (512 → 1)
| 
▼
(N,1,30,30)
```

The discriminator is applied to the VAE-generated image reconstruction to encourage the VAE to generate outputs that
fool the discriminator. E.g. the VAE wants `logits_fake` to be positive, which indicates that the discriminator thinks the reconstructed image is real.
```python
logits_fake = discriminator(reconstructions)
g_loss = -torch.mean(logits_fake)
```
The discriminator loss is balanced with the l2-reconstruction and lpips loss by rescaling according to the gradient norms.
```python
nll_loss = rec_loss + p_scale * p_loss`
nll_grads = grad(nll_loss, last_layer)
g_grads = grad(g_loss, last_layer)
disc_weight = norm(nll_grads) / (norm(g_grads) + 1e-4)
```
#### Final Loss
```python
p_scale = 0.5
kl_scale = 1e-6
loss = rec_loss + p_scale * p_loss + kl_scale * kl_loss + disc_weight * g_loss
```
### Training the Discriminator
- Discriminator training only begins after 50k iterations
- On alternating steps, either the VAE is updated or the discriminator is updated.
- The discriminator to trained to distinguish real from reconstructed images.
```python
logits_real = discriminator(targets)
logits_fake = discriminator(reconstructions)
d_loss = disc_loss(logits_real, logits_fake)
```
- Uses either hinge loss or vanilla GAN loss.

**Hinge Loss**
-  commonly used in modern GANs (e.g., BigGAN, StyleGAN2). It's known for improved training stability and sharper outputs.
-  Encourages margin between real and fake logits


<img src="https://github.com/user-attachments/assets/cd52bad3-abae-4fb4-8be6-0f7112876d96" width="500"/>


**Vanilla GAN loss**
- original GAN loss proposed in the Goodfellow et al. paper (2014)
- Encourages D(x) → 1 for real, D(𝑥̃) → 0 for fake.

<img src="https://github.com/user-attachments/assets/95d46a50-66b7-47f3-9a32-aaa7122a2740" width="450"/>

| Loss Type       | Discriminator Loss                                                  | Generator Loss                          | Discriminator Output |
|------------------|---------------------------------------------------------------------|------------------------------------------|------------------------|
| **Hinge**         | `max(0, 1 - D(x_real)) + max(0, 1 + D(x_fake))`                    | `-D(x_fake)`                             | Raw scores (logits)    |
| **Vanilla GAN**   | `-log(D(x_real)) - log(1 - D(x_fake))`                             | `-log(D(x_fake))`                        | Probabilities (0–1)    |

## Training the VAE
The VAE can be trained using the [train_autoencoder.py](ldm/tools/train_autoencoder.py) script that was adopted from [diffusers/autoencoderkl](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/autoencoderkl).

The exact training recipe for the `KL F-8 VAE` from the [Latent Diffusion paper](https://arxiv.org/abs/2112.10752) is unclear, 
but we suspect it was trained
- on [OpenImages dataset](https://storage.googleapis.com/openimages/web/index.html), which is a very large labeled image dataset, that has since increased since the LDM paper publication. 
- with batch size = 192 on 16x V100 GPUs for ~250k iterations

The original `KL F-8 VAE` was later fine-tuned with a 1:1 ratio on 
[LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/) and  LAION-Humans datasets and 
published to [Huggingface sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse).
The training batch size = 192 on 16x A100s (batch size = 12 per GPU)

OpenImages V7 can be [downloaded from FiftyOne](https://docs.voxel51.com/dataset_zoo/datasets.html#open-images-v7)
