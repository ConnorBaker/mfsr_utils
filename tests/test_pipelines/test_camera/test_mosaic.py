from typing import Callable, Literal, cast, get_args

import pytest
import torch
from hypothesis import given
from hypothesis_torch_utils.strategies.sized_n3hw_tensors import sized_n3hw_tensors
from torch import Tensor

from mfsr_utils.pipelines.camera import demosaic, mosaic
from tests.utils import (
    DeviceName,
    FloatDtypeName,
    TensorInvariantFnName,
    get_device,
    get_float_dtype,
    get_tensor_invariant_fn,
    parametrize_device_name_float_dtype_name,
    parametrize_tensor_invariant_fn_name,
)


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


MosaicFnTy = Callable[[Tensor], Tensor]
MosaicFnName = Literal["mosaic", "compiled_mosaic"]
compiled_mosaic = cast(MosaicFnTy, torch.compile(mosaic))  # type: ignore
parametrize_mosaic_fn_name = pytest.mark.parametrize("mosaic_fn_name", get_args(MosaicFnName))


def get_mosaic_fn(mosaic_fn_name: MosaicFnName) -> MosaicFnTy:
    match mosaic_fn_name:
        case "mosaic":
            return mosaic
        case "compiled_mosaic":
            return compiled_mosaic  # type: ignore


@parametrize_device_name_float_dtype_name
@parametrize_tensor_invariant_fn_name
@parametrize_mosaic_fn_name
def test_mosaic_tensor_invariant(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    tensor_invariant_fn_name: TensorInvariantFnName,
    mosaic_fn_name: MosaicFnName,
) -> None:
    """
    Tests that mosaic preserves a given tensor invariant.

    Args:
        device_name: The name of the device to run the test on
        float_dtype_name: The name of the floating dtype to use
        tensor_invariant_fn_name: The name of the tensor invariant to test
        mosaic_fn_name: The name of the mosaic function to test
    """
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    invariant_fn = get_tensor_invariant_fn(tensor_invariant_fn_name)
    mosaic_fn = get_mosaic_fn(mosaic_fn_name)
    search_strategy = sized_n3hw_tensors(device=device, dtype=dtype)

    @given(image=search_strategy)
    def test(image: Tensor) -> None:
        old_shape = list(image.shape)
        old_shape[-3] = 4
        old_shape[-2] //= 2
        old_shape[-1] //= 2
        expected = torch.empty(old_shape, dtype=image.dtype, device=image.device)
        actual = mosaic_fn(image)
        invariant_holds = invariant_fn(expected, actual)
        assert invariant_holds

    test()


@parametrize_device_name_float_dtype_name
@parametrize_tensor_invariant_fn_name
@parametrize_mosaic_fn_name
def test_demosaic_tensor_invariant(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    tensor_invariant_fn_name: TensorInvariantFnName,
    mosaic_fn_name: MosaicFnName,
) -> None:
    """
    Tests that demosaic preserves a given tensor invariant.

    Args:
        device_name: The name of the device to run the test on
        float_dtype_name: The name of the floating dtype to use
        tensor_invariant_fn_name: The name of the tensor invariant to test
        mosaic_fn_name: The name of the mosaic function to test
    """
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    invariant_fn = get_tensor_invariant_fn(tensor_invariant_fn_name)
    mosaic_fn = get_mosaic_fn(mosaic_fn_name)
    search_strategy = sized_n3hw_tensors(device=device, dtype=dtype)

    @given(image=search_strategy)
    def test(image: Tensor) -> None:
        mosaiced_image = mosaic_fn(image)
        actual = demosaic(mosaiced_image)

        invariant_holds = invariant_fn(image, actual)
        assert invariant_holds

    test()


@parametrize_device_name_float_dtype_name
@parametrize_mosaic_fn_name
def test_mosaic_values_match_mosaic_reference_values(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    mosaic_fn_name: MosaicFnName,
) -> None:
    """
    Tests that the mosaiced image has the same values as the original image.

    Args:
        device_name: The name of the device to run the test on
        float_dtype_name: The name of the floating dtype to use
        mosaic_fn_name: The name of the mosaic function to test
    """
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    mosaic_fn = get_mosaic_fn(mosaic_fn_name)
    search_strategy = sized_n3hw_tensors(device=device, dtype=dtype)

    @given(image=search_strategy)
    def test(image: Tensor) -> None:
        expected = _mosaic_reference(image)
        actual = mosaic_fn(image)

        assert actual.allclose(expected)

    test()
