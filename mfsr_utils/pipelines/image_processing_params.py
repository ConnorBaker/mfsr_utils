from typing import Literal, overload

import torch
from torch import Tensor

from mfsr_utils.pipelines import camera
from mfsr_utils.pipelines.noise import Noise
from mfsr_utils.pipelines.rgb_gain import RgbGain


class ImageProcessingParams:
    """Dataclass for storing image processing parameters"""

    rgb2cam: Tensor
    cam2rgb: Tensor
    gain: RgbGain
    noise: Noise

    smoothstep: bool
    compress_gamma: bool

    @overload
    def __init__(
        self,
        rgb2cam: None = None,
        cam2rgb: None = None,
        gain: None | RgbGain = None,
        noise: None | Noise = None,
        smoothstep: bool = True,
        compress_gamma: bool = True,
        random_ccm: Literal[True] = True,
        random_gain: bool = True,
        random_noise: bool = True,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        rgb2cam: None | Tensor = None,
        cam2rgb: None | Tensor = None,
        gain: None = None,
        noise: None | Noise = None,
        smoothstep: bool = True,
        compress_gamma: bool = True,
        random_ccm: bool = True,
        random_gain: Literal[True] = True,
        random_noise: bool = True,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        rgb2cam: None | Tensor = None,
        cam2rgb: None | Tensor = None,
        gain: None | RgbGain = None,
        noise: None = None,
        smoothstep: bool = True,
        compress_gamma: bool = True,
        random_ccm: bool = True,
        random_gain: bool = True,
        random_noise: Literal[True] = True,
    ) -> None:
        ...

    def __init__(
        self,
        rgb2cam: None | Tensor = None,
        cam2rgb: None | Tensor = None,
        gain: None | RgbGain = None,
        noise: None | Noise = None,
        smoothstep: bool = True,
        compress_gamma: bool = True,
        random_ccm: bool = True,
        random_gain: bool = True,
        random_noise: bool = True,
    ) -> None:
        if rgb2cam is None or cam2rgb is None:
            rgb2cam = camera.random_ccm() if random_ccm else torch.eye(3)
            cam2rgb = rgb2cam.inverse()

        self.rgb2cam = rgb2cam
        self.cam2rgb = cam2rgb

        if gain is None:
            gain = RgbGain.random() if random_gain else RgbGain(1.0, 1.0, 1.0)

        self.gain = gain

        if noise is None:
            noise = Noise.random() if random_noise else Noise(0.0, 0.0)

        self.noise = noise

        self.smoothstep = smoothstep
        self.compress_gamma = compress_gamma

        return
