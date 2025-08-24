import torch
from torch import nn
from typing import Union, Optional
from dataclasses import dataclass
"""
{'_class_name': 'PNDMScheduler', '_diffusers_version': '0.7.0.dev0', 
'beta_end': 0.012, 'beta_schedule': 'scaled_linear', 'beta_start': 0.00085,
'num_train_timesteps': 1000, 'set_alpha_to_one': False,
'skip_prk_steps': True, 'steps_offset': 1, 'trained_betas': None, 'clip_sample': False}

"""

@dataclass
class DDPMSchedulerOutput():
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None



class DDPMScheduler():
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_schedule: str = "linear",
    ):
        self.prediction_type = "epsilon" # "v_prediction"

        self.init_noise_sigma = 1.0 # match training scale
        self.timesteps = None

        # Noise schedule beta and alpha values
        self.betas = get_beta_schedule(beta_schedule, num_train_timesteps).to(dtype=torch.float32) # (num_timesteps,)
        self.num_train_timesteps = int(self.betas.shape[0])
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # alpha_bar_t

        # Other coefficients needed to transform between x_t, x_0, and noise
        # Note that according to Eq. (4) and its reparameterization in Eq. (14),
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # where noise is sampled from N(0, 1)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) # sqrt(alpha_bar_t)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For posterior q(x_{t-1} | x_t, x_0) according to Eq. (6) and (7) of the paper.
        # alpha_bar_{t-1}
        alphas_cumprod_prev = nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

        posterior_var = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_std = torch.sqrt(posterior_var.clamp(min=1e-20))


    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Add noise to the latents according to the noise magnitude at each timestep forward
        """
        return noisy_samples

    def set_timestamps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """
        set the .timesteps array from 1 to slef.num_train_timesteps
        """
        pass

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    def step(
        self,
        model_output: torch.Tensor, # eps
        timestep: int,
        sample: torch.Tensor, # latents
    ):
        """
        Return some structure that has attribute .prev_sample
        """
        pass


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_beta_schedule(beta_schedule, timesteps):
    if beta_schedule == "linear":
        beta_schedule_fn = linear_beta_schedule
    elif beta_schedule == "cosine":
        beta_schedule_fn = cosine_beta_schedule
    elif beta_schedule == "sigmoid":
        beta_schedule_fn = sigmoid_beta_schedule
    else:
        raise ValueError(f"unknown beta schedule {beta_schedule}")

    betas = beta_schedule_fn(timesteps)
    return betas
