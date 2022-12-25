from typing import Literal

import torch
from hypothesis import given
from hypothesis_torch_utils.strategies.sized_n3hw_tensors import sized_n3hw_tensors
from torch import Tensor

from mfsr_utils.pipelines.camera import demosaic, mosaic


def _mosaic_reference(image: Tensor, mode: Literal["grbg", "rggb"] = "rggb") -> Tensor:
    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape
    if image.dim() == 3:
        image = image.unsqueeze(0)

    if mode == "rggb":
        red = image[:, 0, 0::2, 0::2]
        green_red = image[:, 1, 0::2, 1::2]
        green_blue = image[:, 1, 1::2, 0::2]
        blue = image[:, 2, 1::2, 1::2]
        image = torch.stack((red, green_red, green_blue, blue), dim=1)
    elif mode == "grbg":
        green_red = image[:, 1, 0::2, 0::2]
        red = image[:, 0, 0::2, 1::2]
        blue = image[:, 2, 0::2, 1::2]
        green_blue = image[:, 1, 1::2, 1::2]
        image = torch.stack((green_red, red, blue, green_blue), dim=1)

    if len(shape) == 3:
        return image.view((4, shape[-2] // 2, shape[-1] // 2))
    else:
        return image.view((-1, 4, shape[-2] // 2, shape[-1] // 2))


# Property-based tests which ensure:
# - The mosaiced image has four channels
# - The mosaiced image is half the height of the original image
# - The mosaiced image is half the width of the original image
# - The mosaiced image has the same dtype as the original image
# - The mosaiced image is on the same device as the original image
# - The demosaiced image has the same shape as the original image
# - The demosaiced image has the same dtype as the original image
# - The demosaiced image is on the same device as the original image


@given(image=sized_n3hw_tensors())
def test_mosaic_has_four_channels(image: Tensor) -> None:
    """
    Tests that the mosaiced image has four channels.

    Args:
        image: A N3HW image of floating dtype
    """
    expected = 4
    _, actual, _, _ = mosaic(image).shape
    assert actual == expected


@given(image=sized_n3hw_tensors())
def test_mosaic_has_half_height(image: Tensor) -> None:
    """
    Tests that the mosaiced image is half the height of the original image.

    Args:
        image: A N3HW image of floating dtype
    """
    expected = image.shape[-2] // 2
    actual = mosaic(image).shape[-2]
    assert actual == expected


@given(image=sized_n3hw_tensors())
def test_mosaic_has_half_width(image: Tensor) -> None:
    """
    Tests that the mosaiced image is half the width of the original image.

    Args:
        image: A N3HW image of floating dtype
    """
    expected = image.shape[-1] // 2
    actual = mosaic(image).shape[-1]
    assert actual == expected


@given(image=sized_n3hw_tensors())
def test_mosaic_dtype_invariance(image: Tensor) -> None:
    """
    Tests that the mosaiced image has the same dtype as the original image.

    Args:
        image: A N3HW image of floating dtype
    """
    expected = image.dtype
    actual = mosaic(image).dtype
    assert actual == expected


@given(image=sized_n3hw_tensors())
def test_mosaic_device_invariance(image: Tensor) -> None:
    """
    Tests that the mosaiced image is on the same device as the original image.

    Args:
        image: A N3HW image of floating dtype
    """
    expected = image.device
    actual = mosaic(image).device
    assert actual == expected


@given(image=sized_n3hw_tensors())
def test_demosaic_shape_invariance(image: Tensor) -> None:
    """
    Tests that the demosaiced image has the same shape as the original image.

    Args:
        image: A N3HW image of floating dtype
    """
    expected = image.shape
    mosaiced_image = mosaic(image)
    actual = demosaic(mosaiced_image).shape
    assert actual == expected


@given(image=sized_n3hw_tensors())
def test_demosaic_dtype_invariance(image: Tensor) -> None:
    """
    Tests that the demosaiced image has the same dtype as the original image.

    Args:
        image: A N3HW image of floating dtype
    """
    expected = image.dtype
    mosaiced_image = mosaic(image)
    actual = demosaic(mosaiced_image).dtype
    assert actual == expected


@given(image=sized_n3hw_tensors())
def test_demosaic_device_invariance(image: Tensor) -> None:
    """
    Tests that the demosaiced image is on the same device as the original image.

    Args:
        image: A N3HW image of floating dtype
    """
    expected = image.device
    mosaiced_image = mosaic(image)
    actual = demosaic(mosaiced_image).device
    assert actual == expected


@given(image=sized_n3hw_tensors())
def test_mosaic_dtype_matches_mosaic_reference_dtype(image: Tensor) -> None:
    """
    Tests that the mosaiced image has the same dtype as the original image.

    Args:
        image: A N3HW image of floating dtype
    """
    expected = _mosaic_reference(image).dtype
    actual = mosaic(image).dtype
    assert actual == expected


@given(image=sized_n3hw_tensors())
def test_mosaic_device_matches_mosaic_reference_device(image: Tensor) -> None:
    """
    Tests that the mosaiced image is on the same device as the original image.

    Args:
        image: A N3HW image of floating dtype
    """
    expected = _mosaic_reference(image).device
    actual = mosaic(image).device
    assert actual == expected


@given(image=sized_n3hw_tensors())
def test_mosaic_shape_matches_mosaic_reference_shape(image: Tensor) -> None:
    """
    Tests that the mosaiced image has the same shape as the original image.

    Args:
        image: A N3HW image of floating dtype
    """
    expected = _mosaic_reference(image).shape
    actual = mosaic(image).shape
    assert actual == expected


@given(image=sized_n3hw_tensors())
def test_mosaic_values_match_mosaic_reference_values(image: Tensor) -> None:
    """
    Tests that the mosaiced image has the same values as the original image.

    Args:
        image: A N3HW image of floating dtype
    """
    expected = _mosaic_reference(image)
    actual = mosaic(image)
    assert torch.allclose(actual, expected)
