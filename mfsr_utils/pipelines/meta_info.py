from dataclasses import dataclass, field
from functools import partial

from torch import Tensor

from mfsr_utils.pipelines.noise import Noise
from mfsr_utils.pipelines.rgb_gain import RgbGain


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
    gain: RgbGain = field(default_factory=partial(RgbGain, 0.0, 0.0, 0.0))
    noise: Noise = field(default_factory=partial(Noise, 0.0, 0.0))
