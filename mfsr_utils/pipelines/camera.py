from __future__ import annotations

import torch
import torch.jit
from torch import Tensor

""" Based on http://timothybrooks.com/tech/unprocessing
Functions for forward and inverse camera pipeline. All functions input a torch float tensor of
shape (c, h, w). Additionally, some also support batch operations, i.e. inputs of shape
(b, c, h, w).
"""


def random_ccm(
    dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")
) -> Tensor:
    """Generates random RGB -> Camera color correction matrices.

    Args:
        dtype: The dtype of the returned tensor.
        device: The device of the returned tensor.

    Returns:
        A 3x3 CCM tensor of dtype `dtype` and device `device`.
    """
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = torch.tensor(
        [
            [
                [1.0234, -0.2969, -0.2266],
                [-0.5625, 1.6328, -0.0469],
                [-0.0703, 0.2188, 0.6406],
            ],
            [
                [0.4913, -0.0541, -0.0202],
                [-0.613, 1.3513, 0.2906],
                [-0.1564, 0.2151, 0.7183],
            ],
            [
                [0.838, -0.263, -0.0639],
                [-0.2887, 1.0725, 0.2496],
                [-0.0627, 0.1427, 0.5438],
            ],
            [
                [0.6596, -0.2079, -0.0562],
                [-0.4782, 1.3016, 0.1933],
                [-0.097, 0.1581, 0.5181],
            ],
        ],
        dtype=dtype,
        device=device,
    )

    num_ccms = len(xyz2cams)

    weights = torch.empty(size=(num_ccms, 1, 1), dtype=dtype, device=device).uniform_(0.0, 1.0)
    weights_sum = weights.sum()
    xyz2cam = (xyz2cams * weights).sum(dim=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=dtype,
        device=device,
    )
    rgb2cam = xyz2cam.mm(rgb2xyz)

    # Normalizes each row.
    rgb2cam /= rgb2cam.sum(dim=-1, keepdim=True)
    return rgb2cam


def apply_smoothstep(image: Tensor) -> Tensor:
    """Apply global tone mapping curve and clamps to [0, 1].

    Args:
        image: Image to apply tone mapping to. Any shape.

    Returns:
        Image with tone mapping applied. Shape unchanged.
    """
    smoothstepped: Tensor = image**2 * (3 - 2 * image)
    clamped: Tensor = smoothstepped.clamp(0.0, 1.0)
    return clamped


def invert_smoothstep(image: Tensor) -> Tensor:
    """Approximately inverts a global tone mapping curve and clamps to [0, 1].

    Args:
        image: Image to invert. Any shape.

    Returns:
        Inverted image. Shape unchanged.
    """
    arc_sin: Tensor = torch.asin(1.0 - 2.0 * image.clamp(0.0, 1.0))
    sin: Tensor = torch.sin(arc_sin / 3.0)
    shifted: Tensor = 0.5 - sin
    clamped: Tensor = shifted.clamp(0.0, 1.0)
    return clamped


def gamma_expansion(image: Tensor) -> Tensor:
    """Converts from gamma to linear space and clamps to [0, 1].

    Args:
        image: Image to expand. Any shape.

    Returns:
        Image in linear space. Shape unchanged.
    """
    # Clamps to prevent numerical instability of gradients near zero.
    pre_clamped: Tensor = image.clamp(1e-8)
    expaneded: Tensor = pre_clamped**2.2
    clamped: Tensor = expaneded.clamp(0.0, 1.0)
    return clamped


def gamma_compression(image: Tensor) -> Tensor:
    """Converts from linear to gammaspace and clamps to [0, 1].

    Args:
        image: Image to compress. Any shape.

    Returns:
        Image in gamma space. Shape unchanged.
    """
    # Clamps to prevent numerical instability of gradients near zero.
    pre_clamped: Tensor = image.clamp(1e-8)
    compressed: Tensor = pre_clamped ** (1.0 / 2.2)
    clamped: Tensor = compressed.clamp(0.0, 1.0)
    return clamped


def apply_ccm(image: Tensor, ccm: Tensor) -> Tensor:
    """Applies a color correction matrix to an image of shape (3, H, W) and clamps the results to
    [0, 1].

    Args:
        image: Image to apply CCM to. Shape (3, H, W).
        ccm: Color correction matrix.

    Returns:
        Image with CCM applied. Shape (3, H, W).
    """
    reshaped = image.reshape(3, -1)
    transformed = ccm.mm(reshaped)
    viewed = transformed.view(image.shape)
    clamped = viewed.clamp(0.0, 1.0)
    return clamped


def mosaic_rgbg(image: Tensor) -> Tensor:
    """
    Converts a single-channel raw image to a mosaic tensor.

    Only supports the "RGBG" color description.

    Args:
        image (Tensor): A single-channel raw image of shape (H, W).

    Returns:
        A tensor of shape (4, H//2, W//2) where H and W are the height and width of image.
    """
    red = image[0::2, 0::2]
    green_red = image[0::2, 1::2]
    blue = image[1::2, 0::2]
    green_blue = image[1::2, 1::2]
    mosaiced = torch.stack((red, green_red, blue, green_blue), dim=0)
    return mosaiced


def mosaic(images: Tensor) -> Tensor:
    """Extracts RGGB Bayer planes from a stack of RGB images of shape (N, 3, H, W) and returns a
    3D tensor of shape (N, 4, H // 2, W // 2).

    Args:
        images: Images to mosaic. Shape (N, 3, H, W).

    Returns:
        Mosaiced image. Shape (N, 4, H // 2, W // 2).
    """
    red = images[:, 0, 0::2, 0::2]
    green_red = images[:, 1, 0::2, 1::2]
    green_blue = images[:, 1, 1::2, 0::2]
    blue = images[:, 2, 1::2, 1::2]
    mosaiced = torch.stack((red, green_red, green_blue, blue), dim=1)
    return mosaiced
