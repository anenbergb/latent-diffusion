from typing import List, Union, Optional
from diffusers.models.embeddings import get_timestep_embedding
import torch
from torch import nn
from dataclasses import dataclass, field
from jaxtyping import Float
from einops import einsum, rearrange


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


class TransformerBlock(nn.Module):
    r"""
    Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon value for the layer normalization.
        feed_forward_mult (`int`, *optional*, defaults to 4): The factor to scale out the feed forward inner dimension. 8/3 is a good default for LLMs.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        dropout=0.0,
        activation_fn: str = "geglu",
        norm_eps: float = 1e-5,
        feed_forward_mult: int = 4,
    ):
        super().__init__()
        attention_bias = False
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=True)

        self.attn1 = Attention(
            query_dim=dim,
            num_heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
        )
        self.attn2 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
        )

        ff_dim = int(feed_forward_mult * dim)
        # make the SwiGLU and GEGLU the only options for activation funtions
        if activation_fn == "geglu":
            act_fn = GEGLU(dim, ff_dim, bias=True)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, ff_dim, bias=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=True),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim, bias=True),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Example sizes:
            hidden_states (N,1204,320)
            encoder_hidden_states (N,77,768)
            timestep: (N,1280)
        """
        x1 = self.attn1(self.norm1(hidden_states)) + hidden_states
        x2 = self.attn2(self.norm2(x1), encoder_hidden_states) + x1
        out = self.feed_forward(x2)
        return out


class Attention(nn.Module):
    r"""
    Cross Attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        num_heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attentions.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
    """

    def __init__(
        self,
        query_dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.inner_dim = dim_head * num_heads

        self.q_proj = Linear(query_dim, self.inner_dim, bias=False)
        self.kv_proj = Linear(self.cross_attention_dim, self.inner_dim * 2, bias=False)

        self.output_proj = nn.Sequential(
            Linear(self.inner_dim, self.inner_dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        in_features: Float[torch.Tensor, " ... query_seq_len query_dim"],
        cross_features: Optional[Float[torch.Tensor, " ... cross_seq_len cross_dim"]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the attention mechanism.

        Returns:
            Float[Tensor, " ... query_seq_len d_out"]: Output tensor after applying attention.
        """
        if cross_features is None:
            cross_features = in_features

        q = self.q_proj(in_features)  # (..., query_seq_len, inner_dim)
        kv = self.kv_proj(cross_features)  # (..., cross_seq_len, inner_dim)
        k, v = kv.split(self.inner_dim, dim=-1)  # (..., cross_seq_len, inner_dim) x 2

        q = rearrange(
            q,
            "batch query_seq_len (num_heads dim_head) -> batch num_heads query_seq_len dim_head",
            dim_head=self.dim_head,
            num_heads=self.num_heads,
        )
        k = rearrange(
            k,
            "batch cross_seq_len (num_heads dim_head) -> batch num_heads cross_seq_len dim_head",
            dim_head=self.dim_head,
            num_heads=self.num_heads,
        )
        v = rearrange(
            v,
            "batch cross_seq_len (num_heads dim_head) -> batch num_heads cross_seq_len dim_head",
            dim_head=self.dim_head,
            num_heads=self.num_heads,
        )

        attention = nn.functional.scaled_dot_product_attention(q, k, v)
        attention = rearrange(
            attention, "batch num_heads query_seq_len dim_head -> batch query_seq_len (num_heads dim_head)"
        )
        # output projection
        out = self.output_proj(attention)
        return out


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        dropout: float = 0.0,
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
            nn.Dropout(dropout),
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


class SwiGLU(nn.Module):
    """
    SwiGLU layer from https://arxiv.org/abs/2002.05202


    dim: int number of input channels
    d_ff: int hidden dimension of the feed forward layer

    GELU = x * phi(x),
        where phi(x) is the cumulative distribution fn. for gaussian dist.
    SiLU = x * sigmoid(x)

    GLU = sigmoid(W1 * x) * (W2 * x)
    GEGLU = GELU(W1 * x) * (W2 * x)
    SwiGLU = SiLU(W1 * x) * (W2 * x)

    The complete FFN versions
        FFN_GEGLU = (GELU(W1 * x) * (W2 * x)) * W3
        FFN_SwiGLU = (SiLU(W1 * x) * (W2 * x)) * W3
    """

    def __init__(
        self,
        dim: int,
        d_ff: int,
        bias: bool = True,
    ):
        super().__init__()
        self.w1_w2 = nn.Linear(dim, 2 * d_ff, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.w1_w2(x)
        res, gate = res.chunk(2, dim=-1)
        return res * self.act(gate)


class GEGLU(nn.Module):
    """
    GEGLU layer from https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        dim: int,
        d_ff: int,
        bias: bool = True,
        approximate: str = "none",  # or tanh
    ):
        super().__init__()
        self.w1_w2 = nn.Linear(dim, 2 * d_ff, bias=bias)
        self.act = nn.GELU(approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.w1_w2(x)
        res, gate = res.chunk(2, dim=-1)
        return res * self.act(gate)
