from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def apply_noise(image: Tensor, shot_noise: float = 0.01, read_noise: float = 0.0005) -> Tensor:
    """Adds noise to an image."""
    variance = image * shot_noise + read_noise
    noise = torch.randn_like(image) * variance.sqrt()
    return image + noise


def random_shot_noise(
    min_shot_noise: float = 0.0001,
    max_shot_noise: float = 0.012,
) -> float:
    """
    Generates random shot noise from a log-uniform distribution.

    Args:
        min_shot_noise: Minimum shot noise.
        max_shot_noise: Maximum shot noise.

    Returns:
        Random shot noise.
    """
    log_min_shot_noise = math.log(min_shot_noise)
    log_max_shot_noise = math.log(max_shot_noise)
    log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = math.exp(log_shot_noise)
    shot_noise = max(shot_noise, min_shot_noise)
    shot_noise = min(shot_noise, max_shot_noise)
    return shot_noise


def random_read_noise(
    shot_noise: float,
    read_noise_slope: float = math.e,
    read_noise_intercept: float = 1.14,
    log_read_noise_mu: float = 0.0,
    log_read_noise_sigma: float = 0.26,
) -> float:
    """
    Generates random read noise from a log-log normal distribution.

    Args:
        shot_noise: Shot noise.
        read_noise_slope: Read noise slope. Defaults to math.e.
        read_noise_intercept: Read noise intercept. Defaults to 1.14.
        log_read_noise_mu: Mean of the log read noise.
        log_read_noise_sigma: Standard deviation of the log read noise.

    Returns:
        Random read noise.
    """
    # Based on the following code:
    # read_noise = math.exp(
    #     (math.log(read_noise_slope) * math.log(shot_noise) + math.log(read_noise_intercept))
    #     + random.gauss(mu=log_read_noise_mu, sigma=log_read_noise_sigma)
    # )

    # Rewriting as a product of exponentials:
    # read_noise = math.exp(
    #     (math.log(read_noise_slope) * math.log(shot_noise) + math.log(read_noise_intercept))
    # ) * math.exp(random.gauss(mu=log_read_noise_mu, sigma=log_read_noise_sigma))

    # Rewriting inside of first exponential:
    # read_noise = math.exp(
    #     math.log(read_noise_intercept * shot_noise**read_noise_slope)
    # ) * math.exp(random.gauss(mu=log_read_noise_mu, sigma=log_read_noise_sigma))

    # Simplifying first exponential:
    read_noise: float = (read_noise_intercept * shot_noise**read_noise_slope) * math.exp(
        random.gauss(mu=log_read_noise_mu, sigma=log_read_noise_sigma)
    )
    read_noise = max(read_noise, 0.0)
    read_noise = min(read_noise, 1.0)
    return read_noise


@dataclass(eq=False)
class Noise(nn.Module):
    shot_noise: float = 0.01
    read_noise: float = 0.0005

    def __post_init__(self) -> None:
        super().__init__()

    @staticmethod
    def random(
        min_shot_noise: float = 0.0001,
        max_shot_noise: float = 0.012,
        read_noise_slope: float = math.e,
        read_noise_intercept: float = 1.14,
        log_read_noise_mu: float = 0.0,
        log_read_noise_sigma: float = 0.26,
    ) -> Noise:
        """Generates random noise."""
        shot_noise = random_shot_noise(min_shot_noise, max_shot_noise)
        read_noise = random_read_noise(
            shot_noise,
            read_noise_slope,
            read_noise_intercept,
            log_read_noise_mu,
            log_read_noise_sigma,
        )
        return Noise(shot_noise, read_noise)

    def forward(self, image: Tensor) -> Tensor:  # type: ignore
        """Applies noise to an image."""
        return apply_noise(image, self.shot_noise, self.read_noise)
