from typing import List, Union, Optional
from diffusers.models.embeddings import get_timestep_embedding
import torch
from torch import nn
from dataclasses import dataclass, field


@dataclass
class UNetConfig:
    attention_head_dim: int = field(default=8)
    # The tuple of output channels for each block
    block_out_channels: List[int] = field(default_factory=lambda: [320, 640, 1280, 1280])
    cross_attention_dim: int = field(default=768)
    in_channels: int = field(default=4)
    layers_per_block: int = field(default=2)
    norm_eps: float = field(default=1e-5)
    norm_num_groups: float = field(default=32)
    out_channels: int = field(default=4)
    # Height and width of input/output sample.
    sample_size: int = field(default=32)


class UNet(nn.Module):
    def __init__(
        self,
        attention_head_dim: int = 8,
        block_out_channels: List[int] = [320, 640, 1280, 1280],
        cross_attention_dim: int = 768,
        in_channels: int = 4,
        layers_per_block: int = 2,
        norm_eps: float = 1e-5,
        norm_num_groups: float = 32,
        out_channels: int = 4,
        # Height and width of input/output sample.
        sample_size: int = 32,
    ):
        super().__init__()
        assert len(block_out_channels) > 0
        self.block_out_channels = block_out_channels
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        time_embed_dim = block_out_channels[0] * 4  # 1280
        self.time_embedding = nn.Sequential(
            nn.Linear(block_out_channels[0], time_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim, bias=True),
        )
        self.down_blocks = nn.ModuleList()

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
    ):
        """
        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
        """
        temb = get_timestep_embed(sample, timestep, self.block_out_channels[0])  # (N,320)
        temb = self.time_embedding(temb)  # (N, 1280)

        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=temb, encoder_hidden_states=encoder_hidden_states
            )


def get_time_embed(sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], num_channels: int) -> torch.Tensor:
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        if isinstance(timestep, float):
            dtype = torch.float64
        else:
            dtype = torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps.unsqueeze(0).to(sample.device)

    timesteps = timesteps.expand(sample.shape[0])
    t_emb = get_timestep_embedding(
        timesteps, num_channels, flip_sin_to_cos=True, downscale_freq_shift=1, scale=1, max_period=10000
    ).to(dtype=sample.dtype)
    return t_emb


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        return


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.visual = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(temb_channels, out_channels, bias=True))
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Example shapes:
            input_tensor (N,320,32,32)
            temb (N,1280)
        """
        visual = self.visual(input_tensor)
        temb = self.time(temb)[:, :, None, None]
        out = self.layers(visual + temb)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        return out + input_tensor
