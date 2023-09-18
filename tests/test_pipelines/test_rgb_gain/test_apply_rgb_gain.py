from typing import Callable, Literal, get_args

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis_torch_utils.strategies.sized_3hw_tensors import sized_3hw_tensors
from torch import Tensor

from mfsr_utils.pipelines.rgb_gain import RgbGain, apply_rgb_gain, invert_rgb_gain
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

RgbGainFnTy = Callable[[Tensor, float, float, float], Tensor]
ApplyRgbGainFnName = Literal["apply_rgb_gain", "rgb_gain_module"]
parametrize_apply_rgb_gain_fn_name = pytest.mark.parametrize("apply_rgb_gain_fn_name", get_args(ApplyRgbGainFnName))
InvertRgbGainFnName = Literal[
    "invert_rgb_gain",
    "rgb_gain_module_invert",
]
parametrize_invert_rgb_gain_fn_name = pytest.mark.parametrize("invert_rgb_gain_fn_name", get_args(InvertRgbGainFnName))
RgbGainFnName = Literal[ApplyRgbGainFnName, InvertRgbGainFnName]
parametrize_rgb_gain_fn_name = pytest.mark.parametrize("rgb_gain_fn_name", get_args(RgbGainFnName))


def get_rgb_gain_fn(rgb_gain_fn_name: RgbGainFnName) -> RgbGainFnTy:
    match rgb_gain_fn_name:
        case "apply_rgb_gain":
            return apply_rgb_gain
        case "rgb_gain_module":
            return lambda image, rgb_gain, red_gain, blue_gain: (
                RgbGain(rgb_gain, red_gain, blue_gain)(image)  # type: ignore[no-any-return]
            )
        case "invert_rgb_gain":
            return invert_rgb_gain
        case "rgb_gain_module_invert":
            return lambda image, rgb_gain, red_gain, blue_gain: RgbGain(rgb_gain, red_gain, blue_gain).invert_rgb_gain(
                image
            )


@parametrize_device_name_float_dtype_name
@parametrize_tensor_invariant_fn_name
@parametrize_rgb_gain_fn_name
def test_rgb_gain_fn_tensor_invariant(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    tensor_invariant_fn_name: TensorInvariantFnName,
    rgb_gain_fn_name: ApplyRgbGainFnName,
) -> None:
    """
    Tests that apply_rgb_gain maintains an invariant.

    Args:
        device_name: A device name
        float_dtype_name: A float dtype name
        tensor_invariant_fn_name: A tensor invariant name
        rgb_gain_fn_name: A function name
    """
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    invariant_fn = get_tensor_invariant_fn(tensor_invariant_fn_name)
    rgb_gain_fn = get_rgb_gain_fn(rgb_gain_fn_name)
    search_strategy = sized_3hw_tensors(device=device, dtype=dtype)

    @given(
        image=search_strategy,
        rgb_gain=st.floats(0.2, 1.4),
        red_gain=st.floats(1.9, 2.4),
        blue_gain=st.floats(1.5, 1.9),
    )
    def test(
        image: Tensor,
        rgb_gain: float,
        red_gain: float,
        blue_gain: float,
    ) -> None:
        actual = rgb_gain_fn(image, rgb_gain, red_gain, blue_gain)
        invariant_holds = invariant_fn(image, actual)
        assert invariant_holds

    test()


@parametrize_device_name_float_dtype_name
@parametrize_rgb_gain_fn_name
def test_apply_noise_has_identity_element(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    rgb_gain_fn_name: ApplyRgbGainFnName,
) -> None:
    """
    Tests that RgbGain(1.0, 1.0, 1.0) is the identity element.

    Args:
        device_name: A device name
        float_dtype_name: A float dtype name
        rgb_gain_fn_name: A function name
    """
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    rgb_gain_fn = get_rgb_gain_fn(rgb_gain_fn_name)
    search_strategy = sized_3hw_tensors(device=device, dtype=dtype)

    @given(image=search_strategy)
    def test(image: Tensor) -> None:
        actual = rgb_gain_fn(image, 1.0, 1.0, 1.0)
        assert actual.allclose(image)

    test()
