from __future__ import annotations

import cv2 as cv  # type: ignore[import]
import numpy as np
import numpy.typing as npt
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


def demosaic(image: Tensor) -> Tensor:
    assert isinstance(image, torch.Tensor)
    image_normed = (image.clamp(0.0, 1.0) * 255).type(torch.uint8)

    if image_normed.dim() == 4:
        num_images = image_normed.shape[0]
        batch_input = True
    else:
        num_images = 1
        batch_input = False
        image_normed = image_normed.unsqueeze(0)

    # Generate single channel input for opencv
    im_sc = torch.zeros(
        (num_images, image_normed.shape[-2] * 2, image_normed.shape[-1] * 2, 1),
        device=image_normed.device,
        dtype=torch.uint8,
    )

    im_sc[:, 0::2, 0::2, 0] = image_normed[:, 0, :, :]
    im_sc[:, 0::2, 1::2, 0] = image_normed[:, 1, :, :]
    im_sc[:, 1::2, 0::2, 0] = image_normed[:, 2, :, :]
    im_sc[:, 1::2, 1::2, 0] = image_normed[:, 3, :, :]

    # We cannot convert a tensor on the GPU to a numpy array, so we need to move it to the CPU
    # first.
    im_sc_np: npt.NDArray[np.uint8] = im_sc.to(device="cpu", dtype=torch.uint8).numpy()

    out: list[Tensor] = [
        (
            torch.from_numpy(cv.cvtColor(im, cv.COLOR_BAYER_BG2RGB)).permute(  # type: ignore
                2, 0, 1
            )
            / 255.0
        )
        for im in im_sc_np
    ]

    if batch_input:
        return torch.stack(out, dim=0).to(image.device).type(image.dtype)
    else:
        return out[0].to(image.device).type(image.dtype)
