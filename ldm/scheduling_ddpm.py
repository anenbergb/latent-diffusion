import torch
from torch import nn
from dataclasses import dataclass
import math


@dataclass
class DDPMSchedulerOutput:
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
    pred_original_sample: torch.Tensor | None = None


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_schedule: str = "linear",
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
        device: torch.device = torch.device("cuda"),
    ):
        """

        Args:
            timestep_spacing (`str`, defaults to `"leading"`):
                The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
                Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
            steps_offset: An offset added to the inference steps, as required by some model families.
        """
        self.device = device
        self.prediction_type = "epsilon"  # "v_prediction"

        # inference parameters
        self.steps_offset = steps_offset
        self.timestep_spacing = timestep_spacing
        self.num_inference_steps = None
        self.timesteps = torch.arange(0, num_train_timesteps, dtype=torch.long).flip(0).to(device)

        self.init_noise_sigma = 1.0  # match training scale

        # Noise schedule beta and alpha values
        self.betas = (
            get_beta_schedule(beta_schedule, num_train_timesteps).to(dtype=torch.float32).to(device)
        )  # (num_timesteps,)
        self.num_train_timesteps = int(self.betas.shape[0])
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # alpha_bar_t

        # Other coefficients needed to transform between x_t, x_0, and noise
        # Note that according to Eq. (4) and its reparameterization in Eq. (14),
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # where noise is sampled from N(0, 1)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # sqrt(alpha_bar_t)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For posterior q(x_{t-1} | x_t, x_0) according to Eq. (6) and (7) of the paper.
        # alpha_bar_{t-1}
        alphas_cumprod_prev = nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

        posterior_var = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_std = torch.sqrt(posterior_var.clamp(min=1e-20))

    @property
    def config(self):
        return self

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Add noise to the latents according to the noise magnitude at each timestep forward
        Sample from q(x_t | x_0) according to Eq. (4) of the paper

        Args:
            x_0 (B,4,32,32)
            noise (B,4,32,32) sampled from N(0,1)
            timesteps (B,)
        Returns
            x_t (B,4,32,32)
        """
        a_sqrt = extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        x_t = a_sqrt * x_start + sigma * noise
        return x_t

    def predict_start_from_noise(self, x_t, t, noise):
        """Get x_start from x_t and noise according to Eq. (14) of the paper.
        Args:
            x_t: (b, *) tensor. Noisy image.
            t: (b,) tensor. Time step.
            noise: (b, *) tensor. Noise from N(0, 1).
        Returns:
            x_start: (b, *) tensor. Starting image.
        """
        a_sqrt = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x_start = (x_t - sigma * noise) / a_sqrt
        return x_start

    def predict_noise_from_start(self, x_t, t, x_start):
        """Get noise from x_t and x_start according to Eq. (14) of the paper.
        Args:
            x_t: (b, *) tensor. Noisy image.
            t: (b,) tensor. Time step.
            x_start: (b, *) tensor. Starting image.
        Returns:
            pred_noise: (b, *) tensor. Predicted noise.
        """
        a_sqrt = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        pred_noise = (x_t - a_sqrt * x_start) / sigma
        return pred_noise

    def set_timesteps(
        self,
        num_inference_steps: int,
        **kwargs,
    ):
        """
        set the .timesteps array from 1 to self.num_train_timesteps

        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.

        """
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )
        self.num_inference_steps = num_inference_steps
        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.timestep_spacing == "linspace":
            timesteps = (
                torch.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
                .round()
                .flip(0)
                .to(dtype=torch.long)
            )
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round().flip(0).to(dtype=torch.long)
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = torch.arange(self.num_train_timesteps, 0, -step_ratio).round().to(dtype=torch.long)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )
        self.timesteps = timesteps.to(self.device)

    def scale_model_input(self, sample: torch.Tensor, timestep: int | None = None) -> torch.Tensor:
        return sample

    def step(
        self,
        pred_noise: torch.Tensor,  # eps
        timestep: int,
        x_t: torch.Tensor,  # latents
    ):
        """
        Sample from p(x_{t-1} | x_t) according to Eq. (6) of the paper. Used only during inference.

        Args:
            pred_noise: (B,4,32,32), diffusion model's predicted noise
            timestep: int, Sampling time step
            x_t (B,4,32,32), noisy image at timestep t.
        Returns:
            DDPMSchedulerOutput:
                prev_sample x_(t-1): (B,4,32,32) image at timestep t-1
                prev_original_sample x_0: (B,4,32,32)
        """
        prev_t = self.previous_timestep(timestep)
        t_batch = torch.full((x_t.shape[0],), timestep, device=x_t.device, dtype=torch.long)
        s_batch = torch.full((x_t.shape[0],), prev_t, device=x_t.device, dtype=torch.long)

        x_start = self.predict_start_from_noise(x_t, t_batch, pred_noise)
        # for normal diffusion where x_start is an image, you'd want to clip or threshold
        # x_start to keep values in valid image range, e.g. [-1,1]

        mean, std = self.q_posterior_between(x_start, x_t, t_batch, s_batch)
        x_s = mean + std * torch.randn_like(mean)

        output = DDPMSchedulerOutput(prev_sample=x_s, pred_original_sample=x_start)
        return output

    def q_posterior(self, x_start, x_t, t):
        """Get the posterior q(x_{t-1} | x_t, x_0) according to Eq. (6) and (7) of the paper.
        Args:
            x_start: (b, *) tensor. Predicted start image.
            x_t: (b, *) tensor. Noisy image.
            t: (b,) tensor. Time step.
        Returns:
            posterior_mean: (b, *) tensor. Mean of the posterior.
            posterior_std: (b, *) tensor. Std of the posterior.
        """
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = c1 * x_start + c2 * x_t
        posterior_std = extract(self.posterior_std, t, x_t.shape)
        return posterior_mean, posterior_std

    def q_posterior_between(self, x_start, x_t, t, s):
        """
        Posterior q(x_s | x_t, x_0) for arbitrary s < t.
        t: (b,) current timestep
        s: (b,) previous timestep you want to jump to (can be t-1, or a skip)

        Very similar to equations 6 and 7 from the paper but incorporating alpha_bar_s

        Returns:
        mean, std with shapes like x_t
        """
        A_t = extract(self.alphas_cumprod, t, x_t.shape)  # alpha_bar_t
        A_s = self._extract_alpha_bar(s, x_t.shape)  # alpha_bar_s (handles s == -1)

        alpha_ts = A_t / A_s
        beta_ts = 1.0 - alpha_ts

        c1 = torch.sqrt(A_s) * beta_ts / (1.0 - A_t)  # coeff for x_0
        c2 = torch.sqrt(alpha_ts) * (1.0 - A_s) / (1.0 - A_t)  # coeff for x_t

        mean = c1 * x_start + c2 * x_t
        var = (1.0 - A_s) * beta_ts / (1.0 - A_t)
        std = torch.sqrt(var.clamp(min=1e-20))
        return mean, std

    def previous_timestep(self, timestep):
        # if we've set a custom number of inference timesteps
        if self.num_inference_steps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            prev_t = timestep - 1
        return prev_t

    def _extract_alpha_bar(self, t, x_shape):
        """
        Like `extract(self.alphas_cumprod, t, x_shape)` but supports t == -1
        by defining \bar{alpha}_{-1} = 1.0 (useful for the very last step).
        """
        b, *_ = t.shape
        # mask for s == -1
        ones = torch.ones(b, *([1] * (len(x_shape) - 1)), device=t.device, dtype=self.alphas_cumprod.dtype)
        valid_mask = t >= 0
        # clamp negatives to 0 just for gather (we'll overwrite with ones via mask)
        t_clamped = torch.where(valid_mask, t, torch.zeros_like(t))
        vals = self.alphas_cumprod.gather(-1, t_clamped).reshape(b, *([1] * (len(x_shape) - 1)))
        return torch.where(valid_mask.view(b, *([1] * (len(x_shape) - 1))), vals, ones)


def extract(a, t, x_shape):
    """
    Extracts the appropriate coefficient values based on the given timesteps.

    This function gathers the values from the coefficient tensor `a` according to
    the given timesteps `t` and reshapes them to match the required shape such that
    it supports broadcasting with the tensor of given shape `x_shape`.

    Args:
        a (torch.Tensor): A tensor of shape (T,), containing coefficient values for all timesteps.
        t (torch.Tensor): A tensor of shape (b,), representing the timesteps for each sample in the batch.
        x_shape (tuple): The shape of the input image tensor, usually (b, c, h, w).

    Returns:
        torch.Tensor: A tensor of shape (b, 1, 1, 1), containing the extracted coefficient values
                      from a for corresponding timestep of each batch element, reshaped accordingly.
    """
    b, *_ = t.shape  # Extract batch size from the timestep tensor
    out = a.gather(-1, t)  # Gather the coefficient values from `a` based on `t`
    out = out.reshape(b, *((1,) * (len(x_shape) - 1)))  # Reshape to (b, 1, 1, 1) for broadcasting
    return out


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
    as proposed in https://arxiv.org/abs/2212.11972
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
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
