from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def apply_rgb_gain(image: Tensor, rgb_gain: float, red_gain: float, blue_gain: float) -> Tensor:
    """Applies gains to an image."""
    assert image.dim() == 3
    channels = image.shape[0]
    assert channels == 3 or channels == 4
    middle = [1.0] * (channels // 2)

    gains = (
        torch.tensor([red_gain, *middle, blue_gain], dtype=image.dtype, device=image.device)
        * rgb_gain
    )
    gains = gains.view(-1, 1, 1)

    return (image * gains).clamp(0.0, 1.0)


def invert_rgb_gain(image: Tensor, rgb_gain: float, red_gain: float, blue_gain: float) -> Tensor:
    """Inverts gains while safely handling saturated pixels."""
    assert image.dim() == 3
    assert image.shape[0] == 3

    gains = (
        torch.tensor(
            [1.0 / red_gain, 1.0, 1.0 / blue_gain],
            dtype=image.dtype,
            device=image.device,
        )
        / rgb_gain
    )
    gains = gains.view(-1, 1, 1)

    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray = image.mean(dim=0, keepdim=True)
    inflection = 0.9
    mask = ((gray - inflection) / (1.0 - inflection)).clamp(0.0) ** 2.0

    safe_gains = torch.max(input=mask + (1.0 - mask) * gains, other=gains)  # type: ignore
    return image * safe_gains


def random_rgb_gain(
    rgb_gain_mu: float = 0.8,
    rgb_gain_sigma: float = 0.1,
    red_gain_min: float = 1.9,
    red_gain_max: float = 2.4,
    blue_gain_min: float = 1.5,
    blue_gain_max: float = 1.9,
) -> tuple[float, float, float]:
    """Generates random gains for brightening and white balance."""
    rgb_gain = 1.0 / random.gauss(rgb_gain_mu, rgb_gain_sigma)
    red_gain = random.uniform(red_gain_min, red_gain_max)
    blue_gain = random.uniform(blue_gain_min, blue_gain_max)
    return rgb_gain, red_gain, blue_gain


@dataclass(eq=False)
class RgbGain(nn.Module):
    """Container for gains.
    RGB gain represents brightening.
    Red and blue gains represent white balance.
    """

    rgb_gain: float
    red_gain: float
    blue_gain: float

    def __post_init__(self) -> None:
        super().__init__()

    @staticmethod
    def random(
        rgb_gain_mu: float = 0.8,
        rgb_gain_sigma: float = 0.1,
        red_gain_min: float = 1.9,
        red_gain_max: float = 2.4,
        blue_gain_min: float = 1.5,
        blue_gain_max: float = 1.9,
    ) -> RgbGain:
        return RgbGain(
            *random_rgb_gain(
                rgb_gain_mu,
                rgb_gain_sigma,
                red_gain_min,
                red_gain_max,
                blue_gain_min,
                blue_gain_max,
            )
        )

    def forward(self, image: Tensor) -> Tensor:  # type: ignore
        """Applies gains to an image."""
        return apply_rgb_gain(image, self.rgb_gain, self.red_gain, self.blue_gain)

    def invert_rgb_gain(self, image: Tensor) -> Tensor:
        """Inverts gains while safely handling saturated pixels."""
        return invert_rgb_gain(image, self.rgb_gain, self.red_gain, self.blue_gain)
