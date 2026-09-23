"""MambaBlocksSequential: Stack of Mamba/BiMamba blocks.

Adapted from xi-j/Mamba-TasNet's mamba_blocks.py, which is itself based on
mamba_ssm's mixer_seq_simple.py (Gu & Dao, 2024).

Stacks N pre-norm Block(Mamba/BiMamba) layers with:
  - Cross-layer residual tracking (Add -> LN -> Mixer pattern)
  - Final layer norm
  - GPT-2 weight initialization (1/sqrt(n_layer) scaling on out_proj)

Requires mamba-ssm library (Linux + CUDA only).
"""

import math
import torch
import torch.nn as nn

from functools import partial

from mamba_ssm import Mamba
from mamba_ssm.modules.block import Block

from .bimamba import Mamba as BiMamba

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
    d_model,
    ssm_cls,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
):
    """Create a single pre-norm Mamba/BiMamba block.

    Uses mamba_ssm.modules.block.Block for the Add -> LN -> Mixer pattern
    with no MLP (mlp_cls=nn.Identity).
    """
    if ssm_cfg is None:
        ssm_cfg = {}

    mixer_cls = partial(ssm_cls, layer_idx=layer_idx, **ssm_cfg)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
    )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls=nn.Identity,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Only used for embedding layer
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    """GPT-2 style weight initialization with prenorm residual rescaling.

    Scales out_proj.weight by 1/sqrt(n_residuals_per_layer * n_layer) to
    account for accumulation on the residual path with model depth.

    Reference: https://openai.com/blog/better-language-models/ (GPT-2)
    """
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the
        #   > residual path with model depth. Scale the weights of residual layers
        #   > at initialization by a factor of 1/sqrt(N) where N is the # of
        #   > residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MambaBlocksSequential(nn.Module):
    """Stack of N Mamba/BiMamba blocks with cross-layer residual tracking.

    Adapted from xi-j/Mamba-TasNet. Each block uses the Add -> LN -> Mixer
    pattern (mamba_ssm.modules.block.Block), passing (hidden_states, residual)
    between layers for efficient prenorm computation.

    Args:
        n_mamba: Number of Mamba blocks to stack.
        bidirectional: Use BiMamba (True) or standard Mamba (False).
        d_model: Input/output feature dimension (bottleneck dimension).
        d_state: SSM state dimension (default 16).
        expand: Inner dimension expansion factor (default 2).
        d_conv: Local convolution width (default 4).
        dt_rank: Rank for delta projection (default "auto").
        conv_bias: Whether conv1d has bias (default True).
        bias: Whether linear projections have bias (default False).
        fused_add_norm: Use fused Triton add+norm kernels (default False).
        rms_norm: Use RMSNorm instead of LayerNorm (default False).
        norm_epsilon: Norm epsilon (default 1e-5).
        residual_in_fp32: Keep residual in fp32 (default False).
        initializer_cfg: Override kwargs for _init_weights.
    """

    def __init__(
        self,
        n_mamba: int,
        bidirectional: bool = False,
        d_model: int = 256,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dt_rank: str = "auto",
        conv_bias: bool = True,
        bias: bool = False,
        fused_add_norm: bool = False,
        rms_norm: bool = False,
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = False,
        initializer_cfg=None,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        if fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError(
                    "fused_add_norm requires Triton LayerNorm / RMSNorm kernels"
                )

        if rms_norm and RMSNorm is None:
            raise ImportError(
                "rms_norm=True requires RMSNorm from mamba_ssm.ops.triton.layer_norm"
            )

        ssm_cls = BiMamba if bidirectional else Mamba
        ssm_cfg = {
            "d_state": d_state,
            "expand": expand,
            "d_conv": d_conv,
            "dt_rank": dt_rank,
            "conv_bias": conv_bias,
            "bias": bias,
        }

        self.layers = nn.ModuleList([
            create_block(
                d_model=d_model,
                ssm_cls=ssm_cls,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
            )
            for i in range(n_mamba)
        ])

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_mamba,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, x, inference_params=None):
        """Forward pass through all blocks with cross-layer residual tracking.

        Args:
            x: [B, L, D] input tensor.
            inference_params: Optional inference cache parameters.

        Returns:
            [B, L, D] output tensor.
        """
        hidden_states = x
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        # Final norm: apply residual + norm after last block
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states
