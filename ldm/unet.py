from typing import List, Union, Optional, Tuple
from diffusers.models.embeddings import get_timestep_embedding
import torch
from torch import nn
from jaxtyping import Float
from einops import einsum, rearrange


class UNet(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 8,
        block_out_channels: List[int] = [320, 640, 1280, 1280],
        cross_attention_dim: int = 768,
        in_channels: int = 4,
        layers_per_block: int = 2,
        transformer_layers_per_block: int = 1,
        norm_num_groups: float = 32,
        dropout: float = 0.0,
        feed_forward_mult: int = 4,
        activation_fn: str = "geglu",
    ):
        super().__init__()
        assert len(block_out_channels) > 1
        self.block_out_channels = block_out_channels
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        temb_channels = block_out_channels[0] * 4  # 1280
        self.time_embedding = nn.Sequential(
            nn.Linear(block_out_channels[0], temb_channels, bias=True),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels, bias=True),
        )

        def get_down_block(i):
            in_chan = block_out_channels[0] if i == 0 else block_out_channels[i - 1]
            # last block is simple ResNet DownBlock
            if i == len(block_out_channels) - 1:
                return DownBlock(
                    in_channels=in_chan,
                    out_channels=block_out_channels[i],
                    temb_channels=temb_channels,
                    dropout=dropout,
                    num_layers=layers_per_block,
                    norm_num_groups=norm_num_groups,
                )
            else:
                return CrossAttnDownBlock(
                    in_channels=in_chan,
                    out_channels=block_out_channels[i],
                    temb_channels=temb_channels,
                    dropout=dropout,
                    num_layers=layers_per_block,
                    transformer_layers_per_block=transformer_layers_per_block,
                    norm_num_groups=norm_num_groups,
                    num_attention_heads=num_attention_heads,
                    cross_attention_dim=cross_attention_dim,
                    feed_forward_mult=feed_forward_mult,
                    activation_fn=activation_fn,
                )

        self.down_blocks = nn.ModuleList([get_down_block(i) for i in range(len(block_out_channels))])
        self.mid_block = CrossAttnBlock(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=1,
            transformer_layers_per_block=transformer_layers_per_block,
            norm_num_groups=norm_num_groups,
            num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim,
            feed_forward_mult=feed_forward_mult,
            activation_fn=activation_fn,
        )

        def get_up_block(i):
            rev_idx = len(block_out_channels) - 1 - i
            in_chan = block_out_channels[-1] if i == 0 else block_out_channels[rev_idx + 1]
            final_skip_channels = block_out_channels[0] if rev_idx == 0 else block_out_channels[rev_idx - 1]

            # all blocks upsample except the final up block
            add_upsample = i < len(block_out_channels) - 1
            if i == 0:
                return UpBlock(
                    in_channels=in_chan,
                    out_channels=block_out_channels[rev_idx],
                    final_skip_channels=final_skip_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    num_layers=layers_per_block + 1,
                    norm_num_groups=norm_num_groups,
                )
            else:
                return CrossAttnUpBlock(
                    in_channels=in_chan,
                    out_channels=block_out_channels[rev_idx],
                    final_skip_channels=final_skip_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    num_layers=layers_per_block + 1,
                    transformer_layers_per_block=transformer_layers_per_block,
                    norm_num_groups=norm_num_groups,
                    num_attention_heads=num_attention_heads,
                    cross_attention_dim=cross_attention_dim,
                    feed_forward_mult=feed_forward_mult,
                    activation_fn=activation_fn,
                    add_upsample=add_upsample,
                )

        self.up_blocks = nn.ModuleList([get_up_block(i) for i in range(len(block_out_channels))])
        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_out_channels[0]),
            nn.SiLU(),
            nn.Conv2d(block_out_channels[0], in_channels, kernel_size=3, stride=1, padding=1),
        )

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
        temb = get_time_embed(sample, timestep, self.block_out_channels[0])  # (N,320)
        temb = self.time_embedding(temb)  # (N, 1280)

        hidden_states = self.conv_in(sample)
        down_block_res_states = (hidden_states,)
        for downsample_block in self.down_blocks:
            if isinstance(downsample_block, CrossAttnDownBlock):
                hidden_states, res_states = downsample_block(hidden_states, temb, encoder_hidden_states)
            else:  # DownBlock
                hidden_states, res_states = downsample_block(hidden_states, temb)
            down_block_res_states += res_states

        hidden_states = self.mid_block(hidden_states, temb, encoder_hidden_states)

        for i, upsample_block in enumerate(self.up_blocks):
            res_states = down_block_res_states[-len(upsample_block.resnets) :]
            down_block_res_states = down_block_res_states[: -len(upsample_block.resnets)]

            if isinstance(upsample_block, CrossAttnUpBlock):
                hidden_states = upsample_block(hidden_states, res_states, temb, encoder_hidden_states)
            else:  # UpBlock
                hidden_states = upsample_block(hidden_states, res_states, temb)

        output = self.final_conv(hidden_states)
        return output


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


class CrossAttnDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 2,
        transformer_layers_per_block: int = 1,
        norm_num_groups: int = 32,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 1280,
        feed_forward_mult: int = 4,
        activation_fn: str = "geglu",
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    dropout=dropout,
                )
            )
            self.attentions.append(
                TransformerModel(
                    in_channels=out_channels,
                    num_attention_heads=num_attention_heads,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    feed_forward_mult=feed_forward_mult,
                    activation_fn=activation_fn,
                )
            )
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            output_states = output_states + (hidden_states,)

        hidden_states = self.downsample(hidden_states)
        output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        final_skip_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            in_chan = in_channels if i == 0 else out_channels
            skip_channels = final_skip_channels if i == num_layers - 1 else out_channels
            self.resnets.append(
                ResnetBlock(
                    in_channels=in_chan + skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    dropout=dropout,
                )
            )
        self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        res_hidden_states_tuple: output activations from the corresponding resnet blocks in the DownBlock
        """
        for i, resnet in enumerate(self.resnets, start=1):
            hidden_states = torch.cat([hidden_states, res_hidden_states_tuple[-i]], dim=1)
            hidden_states = resnet(hidden_states, temb)

        hidden_states = nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.upsample(hidden_states)
        return hidden_states


class CrossAttnUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        final_skip_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        norm_num_groups: int = 32,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 1280,
        feed_forward_mult: int = 4,
        activation_fn: str = "geglu",
        add_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(num_layers):
            in_chan = in_channels if i == 0 else out_channels
            skip_channels = final_skip_channels if i == num_layers - 1 else out_channels
            self.resnets.append(
                ResnetBlock(
                    in_channels=in_chan + skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    dropout=dropout,
                )
            )
            self.attentions.append(
                TransformerModel(
                    in_channels=out_channels,
                    num_attention_heads=num_attention_heads,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    feed_forward_mult=feed_forward_mult,
                    activation_fn=activation_fn,
                )
            )
        self.upsample = None
        if add_upsample:
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions), start=1):
            hidden_states = torch.cat([hidden_states, res_hidden_states_tuple[-i]], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        if self.upsample:
            hidden_states = nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            hidden_states = self.upsample(hidden_states)
        return hidden_states


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        norm_num_groups: int = 32,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 1280,
        feed_forward_mult: int = 4,
        activation_fn: str = "geglu",
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    dropout=dropout,
                )
                for i in range(num_layers + 1)
            ]
        )
        self.attentions = nn.ModuleList(
            [
                TransformerModel(
                    in_channels=out_channels,
                    num_attention_heads=num_attention_heads,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    feed_forward_mult=feed_forward_mult,
                    activation_fn=activation_fn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class TransformerModel(nn.Module):
    """

    Parameters:
        in_channels (`int`):
            The number of channels in the input and output
        num_attention_heads (`int`, defaults to 16): The number of heads to use for multi-head attention.
        num_layers (`int`, defaults to 1): The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`): The number of `encoder_hidden_states` dimensions to use.
        dropout (`float`, defaults to 0.0): The dropout probability to use.
        norm_num_groups (`int`)
        feed_forward_mult (`int`, defaults to 4): The factor to scale out the feed forward inner dimension. 8/3 is a good default for LLMs.
        activation_fn (`str`, defaults to `"geglu"`): Activation function to use in feed-forward.
    """

    def __init__(
        self,
        in_channels: int = 320,
        num_attention_heads: int = 16,
        num_layers: int = 1,
        cross_attention_dim: int = 768,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        feed_forward_mult: int = 4,
        activation_fn: str = "geglu",
    ):
        super().__init__()
        assert in_channels % num_attention_heads == 0, "in_channels must be divisible by num_attention_heads"
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, affine=True)
        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    in_channels=in_channels,
                    num_attention_heads=num_attention_heads,
                    cross_attention_dim=cross_attention_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    feed_forward_mult=feed_forward_mult,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ):
        """
        Example shape:
            hidden_states (N,320,32,32)
            encoder_hidden_states (N,77,768)
        """
        height, width = hidden_states.shape[-2:]
        x = self.proj_in(self.norm(hidden_states))
        x = rearrange(
            x,
            "batch dim height width -> batch (height width) dim",
        )
        for block in self.transformer_blocks:
            x = block(x, encoder_hidden_states)
        x = rearrange(
            x,
            "batch (height width) dim -> batch dim height width",
            height=height,
            width=width,
        )
        return self.proj_out(x)


class TransformerBlock(nn.Module):
    r"""
    Transformer block.

    Parameters:
        in_channels (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        cross_attention_dim (`int`): The size of the encoder_hidden_states vector for cross attention.
        dropout (`float`, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, defaults to `"geglu"`): Activation function to be used in feed-forward.
        feed_forward_mult (`int`, defaults to 4): The factor to scale out the feed forward inner dimension. 8/3 is a good default for LLMs.
    """

    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        cross_attention_dim: int,
        dropout=0.0,
        activation_fn: str = "geglu",
        feed_forward_mult: int = 4,
    ):
        super().__init__()
        assert in_channels % num_attention_heads == 0, "in_channels must be divisible by num_attention_heads"
        self.attention_head_dim = in_channels // num_attention_heads

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.attn1 = Attention(
            query_dim=in_channels,
            num_heads=num_attention_heads,
            dim_head=self.attention_head_dim,
            dropout=dropout,
        )
        self.attn2 = Attention(
            query_dim=in_channels,
            dim_head=self.attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
        )

        ff_dim = int(feed_forward_mult * in_channels)
        # make the SwiGLU and GEGLU the only options for activation funtions
        if activation_fn == "geglu":
            act_fn = GEGLU(in_channels, ff_dim, bias=True)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(in_channels, ff_dim, bias=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(in_channels, elementwise_affine=True),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(ff_dim, in_channels, bias=True),
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
        num_heads (`int`,, defaults to 8):
            The number of heads to use for multi-head attentions.
        dim_head (`int`,, defaults to 64):
            The number of channels in each head.
        dropout (`float`, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (`int`):
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

        self.q_proj = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.kv_proj = nn.Linear(self.cross_attention_dim, self.inner_dim * 2, bias=False)

        self.output_proj = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim, bias=True),
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
        norm_num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.visual = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(temb_channels, out_channels, bias=True))
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
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


if __name__ == "__main__":
    print("Creating UNet model...")
    unet = UNet()

    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")

    # Create test tensors
    batch_size = 2
    height, width = 32, 32  # Assuming 256x256 input divided by 8 (VAE downsampling)
    in_channels = 4  # Latent space channels
    seq_len = 77  # CLIP text encoder sequence length
    text_embed_dim = 768  # CLIP text embedding dimension

    print("Creating test tensors...")

    # Sample tensor (latent representation)
    sample = torch.randn(batch_size, in_channels, height, width)
    print(f"Sample shape: {sample.shape}")

    # Timestep (can be tensor, float, or int)
    timestep = torch.randint(0, 1000, (batch_size,))
    print(f"Timestep shape: {timestep.shape}")

    # Text encoder hidden states
    encoder_hidden_states = torch.randn(batch_size, seq_len, text_embed_dim)
    print(f"Encoder hidden states shape: {encoder_hidden_states.shape}")

    print("\nRunning forward pass...")
    try:
        with torch.no_grad():  # Disable gradients for inference
            output = unet(sample, timestep, encoder_hidden_states)
        print(f"Success! Output shape: {output.shape}")
        print(f"Output min: {output.min().item():.4f}, max: {output.max().item():.4f}")
        print(f"Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")

        # Verify output shape matches input
        assert output.shape == sample.shape, f"Output shape {output.shape} doesn't match input shape {sample.shape}"
        print("✓ Output shape verification passed!")

    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback

        traceback.print_exc()

    print("\nTesting different timestep formats...")

    # Test with single float timestep
    try:
        with torch.no_grad():
            output = unet(sample, 500.0, encoder_hidden_states)
        print("✓ Float timestep test passed!")
    except Exception as e:
        print(f"Float timestep test failed: {e}")

    # Test with single int timestep
    try:
        with torch.no_grad():
            output = unet(sample, 250, encoder_hidden_states)
        print("✓ Int timestep test passed!")
    except Exception as e:
        print(f"Int timestep test failed: {e}")

    # Test with different batch sizes
    print("\nTesting different batch sizes...")
    for bs in [1, 3]:
        try:
            test_sample = torch.randn(bs, in_channels, height, width)
            test_timestep = torch.randint(0, 1000, (bs,))
            test_encoder = torch.randn(bs, seq_len, text_embed_dim)

            with torch.no_grad():
                output = unet(test_sample, test_timestep, test_encoder)
            print(f"✓ Batch size {bs} test passed! Output shape: {output.shape}")
        except Exception as e:
            print(f"Batch size {bs} test failed: {e}")

    # Test CUDA if available
    if torch.cuda.is_available():
        print(f"\nCUDA is available! Testing on GPU...")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")

        # Test different precisions
        precisions = [
            (torch.float32, "float32"),
            (torch.bfloat16, "bfloat16"),
        ]

        for dtype, dtype_name in precisions:
            print(f"\nTesting CUDA with {dtype_name} precision...")
            try:
                # Move model to CUDA and set precision
                unet_cuda = unet.to("cuda").to(dtype)

                # Create CUDA tensors with appropriate dtype
                sample_cuda = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=dtype)
                timestep_cuda = torch.randint(0, 1000, (batch_size,), device="cuda")
                encoder_cuda = torch.randn(batch_size, seq_len, text_embed_dim, device="cuda", dtype=dtype)

                print(f"  Sample tensor: {sample_cuda.shape} {sample_cuda.dtype} on {sample_cuda.device}")
                print(f"  Timestep tensor: {timestep_cuda.shape} {timestep_cuda.dtype} on {timestep_cuda.device}")
                print(f"  Encoder tensor: {encoder_cuda.shape} {encoder_cuda.dtype} on {encoder_cuda.device}")

                # Run forward pass
                with torch.no_grad():
                    output_cuda = unet_cuda(sample_cuda, timestep_cuda, encoder_cuda)

                print(f"  ✓ CUDA {dtype_name} test passed! Output shape: {output_cuda.shape}")
                print(f"  Output dtype: {output_cuda.dtype}, device: {output_cuda.device}")
                print(f"  Output stats - min: {output_cuda.min().item():.4f}, max: {output_cuda.max().item():.4f}")
                print(f"  Output stats - mean: {output_cuda.mean().item():.4f}, std: {output_cuda.std().item():.4f}")

                # Test memory usage
                allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                cached_memory = torch.cuda.memory_reserved() / 1024**3  # GB
                print(f"  GPU Memory - Allocated: {allocated_memory:.2f} GB, Cached: {cached_memory:.2f} GB")

                # Verify output shape
                assert output_cuda.shape == sample_cuda.shape, (
                    f"CUDA output shape mismatch: {output_cuda.shape} vs {sample_cuda.shape}"
                )
                print(f"  ✓ CUDA {dtype_name} shape verification passed!")

                # Test different batch sizes on CUDA
                print(f"  Testing different batch sizes on CUDA {dtype_name}...")
                for bs in [1, 4]:
                    try:
                        test_sample_cuda = torch.randn(bs, in_channels, height, width, device="cuda", dtype=dtype)
                        test_timestep_cuda = torch.randint(0, 1000, (bs,), device="cuda")
                        test_encoder_cuda = torch.randn(bs, seq_len, text_embed_dim, device="cuda", dtype=dtype)

                        with torch.no_grad():
                            test_output_cuda = unet_cuda(test_sample_cuda, test_timestep_cuda, test_encoder_cuda)
                        print(f"    ✓ CUDA {dtype_name} batch size {bs} test passed! Shape: {test_output_cuda.shape}")
                    except Exception as e:
                        print(f"    CUDA {dtype_name} batch size {bs} test failed: {e}")

                # Clear GPU memory
                del sample_cuda, timestep_cuda, encoder_cuda, output_cuda
                if "test_sample_cuda" in locals():
                    del test_sample_cuda, test_timestep_cuda, test_encoder_cuda, test_output_cuda
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  CUDA {dtype_name} test failed: {e}")
                import traceback

                traceback.print_exc()

                # Clear GPU memory on failure
                torch.cuda.empty_cache()

        # Move model back to CPU
        try:
            unet = unet.to("cpu").to(torch.float32)
            print(f"\n✓ Model moved back to CPU successfully")
        except Exception as e:
            print(f"Warning: Failed to move model back to CPU: {e}")

    else:
        print(f"\nCUDA is not available. Skipping GPU tests.")
        print("To enable CUDA testing, ensure you have:")
        print("  1. A CUDA-compatible GPU")
        print("  2. CUDA drivers installed")
        print("  3. PyTorch compiled with CUDA support")

    print("\nAll tests completed!")
