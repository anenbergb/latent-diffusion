# CLIP for Latent Diffusion Training

CLIP (Contrastive Language–Image Pretraining) is a dual-encoder model developed by OpenAI that learns to connect images and text through large-scale
contrastive training. The original publication in 2021 was [Learning Transferable Visual Models From Natural Language Supervision
](https://arxiv.org/abs/2103.00020).

- It learns without labeled classes: no ImageNet labels, just web pairs
- Can be used for zero-shot image classification, text-to-image search, or conditioning generative models like LDMs

![image](https://github.com/user-attachments/assets/c32bd061-79b0-43c8-9834-48049fe35156)

#### Text Encoder:
A Transformer that maps tokenized text prompts to a dense vector.

- Architecture: Transformer (e.g., ViT-B, ViT-L, ViT-H for OpenCLIP)
- Tokenizer: Byte-Pair Encoding (BPE)
- Inputs: Prompts like "a photo of a cat"
- Output: A fixed-size embedding vector (e.g., 512, 768, or 1024)
#### Image Encoder
A Vision Transformer (ViT) or ResNet that embeds images into the same vector space.

- Architecture: Either a Vision Transformer (ViT) or ResNet
- Inputs: Images resized to 224×224 (typically)
- Output: A fixed-size embedding vector (same dimension as text encoder)

#### CLIP Training

CLIP is trained on web-scale image–text pairs.
During training, the encoders map their respective inputs to the same embedding space, allowing for cross-modal similarity comparison.
CLIP is trained using a contrastive loss to align text and image pairs.

The CLIP contrastive loss is designed to train the model to associate corresponding image and text pairs, while pushing apart mismatched pairs in the embedding space.
It does this by comparing all text-image combinations in a batch using cosine similarity, and then computing a symmetric cross-entropy loss over the resulting similarity scores.

#### Training Setup:
Given a batch of **N image-text pairs**:
`{(I₁, T₁), (I₂, T₂), ..., (Iₙ, Tₙ)}`


1. Compute the embeddings:

- `image_features = f(I₁, ..., Iₙ)` → shape: `[N, D]`
- `text_features = g(T₁, ..., Tₙ)` → shape: `[N, D]`

2. Normalize both embeddings: unit vectors (L2 norm)

$\hat{f}(I_i) = \frac{f(I_i)}{\|f(I_i)\|}$

$\hat{g}(T_i) = \frac{g(T_i)}{\|g(T_i)\|}$

3. Compute the cosine similarity matrix:

$S_{ij} = \hat{f}(I_i)^\top \hat{g}(T_j)$

This gives a matrix $S \in \mathbb{R}^{N \times N}$

#### Symmetric Contrastive Loss
- For each image, the correct caption should be the most similar text embedding (and vice versa)
- Contrastive loss encourages matched pairs to have high cosine similarity and unmatched pairs to be pushed apart

CLIP uses a **symmetric cross-entropy loss**, which includes:
- Image-to-text loss
- Text-to-image loss

![image](https://github.com/user-attachments/assets/2d71f272-d8d8-44ed-810a-9d8af029f742)

![image](https://github.com/user-attachments/assets/1fe8ee9c-262c-4f28-979c-0c89b7f6a1cb)

- Each cross-entropy treats the correct match (diagonal of the matrix) as the ground truth.
- Encourages matched pairs to have high similarity and unmatched pairs to be pushed apart.

#### Temperature Scaling

CLIP introduces a learnable temperature parameter tau $\tau$, initialized around 0.07:

$$
S_{ij} = \frac{\hat{f}(I_i)^\top \hat{g}(T_j)}{\tau}
$$

- This sharpens or softens the softmax distribution.
- A smaller \( \tau \) emphasizes hard negatives more strongly.
- \( \tau \) is trained as a scalar parameter.


#### CLIP Summary
CLIP 
- Uses all \( N^2 \) combinations per batch — **no need for hard negative mining**.
- **Symmetric**: treats images and texts equally.
- **Efficient** and scalable to large web datasets (e.g., LAION-400M, LAION-2B).
- Learns a **joint embedding space** where images and text are directly comparable.

In each batch, CLIP learns to answer:  
- “Which of these N captions best describes this image?”
- and “Which of these N images best matches this caption?”


## OpenCLIP

[OpenCLIP](https://github.com/mlfoundations/open_clip) is an open-source reproduction and extension of OpenAI’s CLIP, developed by the LAION community. It offers:

- Larger and more diverse training datasets (e.g., LAION-400M, LAION-2B, LAION-5B).
- Wider model variants like ViT-B, ViT-L, ViT-H.
- Public weights and transparent training logs.

#### OpenCLIP ViT-B/16 vs ViT-L/14 vs ViT-H/14

| Feature                        | ViT-B/16           | ViT-L/14            | ViT-H/14                   |
|-------------------------------|--------------------|---------------------|----------------------------|
| **Patch Size**                | 16×16              | 14×14               | 14×14                      |
| **Hidden Size**               | 768                | 1024                | 1280                       |
| **Transformer Blocks**        | 12                 | 24                  | 32                         |
| **MLP Dim**                   | 3072               | 4096                | 5120                       |
| **Heads**                     | 12                 | 16                  | 20                         |
| **Parameters (Text Encoder)** | ~86M               | ~160M               | ~316M                      |
| **Total CLIP Parameters**     | ~151M              | ~428M               | ~1.1B                      |
| **Token Length**              | 77                 | 77                  | 77                         |
| **Embedding Dim**             | 512                | 768                 | 1024                       |
| **Speed (relative)**          | 🟢 Fastest         | 🟡 Medium           | 🔴 Slowest                 |
| **GPU Memory (512×512 input)**| ~1.4GB (float32)   | ~2.8GB              | ~5.2GB                     |
| **Used in SD version**        | –                  | SD v1.4 / v1.5 (OpenAI) | SD v2.0 / v2.1 (OpenCLIP) |


### Why Use OpenCLIP ViT-L/14?

The [`openclip/ViT-L-14`](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) model is a better choice than the original 
[`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14)
because I plan to train the 
Latent Diffusion Model (LDM) on LAION-based datasets such as LAION Aesthetic and LAION Pop.
- Trained on LAION: OpenCLIP ViT-L/14 was trained on the same data distribution I'll be using for LDM training, improving alignment and caption fidelity.
- Open weights and tokenizer: Easily reproducible and adaptable for your training pipelines.
- Improved performance: Empirical results show better or comparable text-image alignment vs OpenAI's original CLIP.

### Stable Diffusion v2 Uses OpenCLIP

Stable Diffusion v2.x models transitioned to using the larger [`OpenCLIP ViT-H/14`](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) model.
This change was made to take advantage of OpenCLIP’s superior performance and LAION alignment.


Given I plan to train a LDM from scratch and use LAION-based datasets, using **OpenCLIP ViT-L/14** ensures better compatibility, data alignment, 
and performance than the original OpenAI CLIP model.
