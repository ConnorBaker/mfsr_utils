from dataclasses import dataclass, field
from functools import partial

from torch import Tensor

from mfsr_utils.pipelines.noises import Noises
from mfsr_utils.pipelines.rgb_gains import RgbGains


@dataclass
class MetaInfo:
    rgb2cam: Tensor
    cam2rgb: Tensor
    smoothstep: bool
    compress_gamma: bool
    norm_factor: float = 1.0
    black_level_subtracted: bool = False
    black_level: None | Tensor = None
    while_balance_applied: bool = False
    cam_wb: None | Tensor = None
    gains: RgbGains = field(default_factory=partial(RgbGains, 0.0, 0.0, 0.0))
    noises: Noises = field(default_factory=partial(Noises, 0.0, 0.0))
