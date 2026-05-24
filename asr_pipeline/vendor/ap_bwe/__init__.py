"""Vendored AP-BWE — minimal copy for inference.

See `README.md` for attribution, license, and the list of modifications
relative to the upstream repository (https://github.com/yxlu-0102/AP-BWE).
"""

from .env import AttrDict
from .model import APNet_BWE_Model
from .stft import amp_pha_istft, amp_pha_stft

__all__ = ["APNet_BWE_Model", "AttrDict", "amp_pha_istft", "amp_pha_stft"]
