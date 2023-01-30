from typing import Callable, Literal, cast, get_args

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis_torch_utils.strategies.sized_3hw_tensors import sized_3hw_tensors
from torch import Tensor

from mfsr_utils.pipelines.noise import Noise, apply_noise
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

ApplyNoiseFnTy = Callable[[Tensor, float, float], Tensor]
ApplyNoiseFnName = Literal[
    "apply_noise", "compiled_apply_noise", "noise_module", "compiled_noise_module"
]
compiled_apply_noise = cast(ApplyNoiseFnTy, torch.compile(apply_noise))  # type: ignore
parametrize_apply_noise_fn_name = pytest.mark.parametrize(
    "apply_noise_fn_name", get_args(ApplyNoiseFnName)
)


def get_apply_noise_fn(apply_noise_fn_name: ApplyNoiseFnName) -> ApplyNoiseFnTy:
    match apply_noise_fn_name:
        case "apply_noise":
            return apply_noise
        case "compiled_apply_noise":
            return compiled_apply_noise
        case "noise_module":
            return lambda image, shot_noise, read_noise: (
                Noise(shot_noise, read_noise)(image)  # type: ignore[no-any-return]
            )
        case "compiled_noise_module":
            return lambda image, shot_noise, read_noise: torch.compile(  # type: ignore
                Noise(shot_noise, read_noise)
            )(image)


@parametrize_device_name_float_dtype_name
@parametrize_tensor_invariant_fn_name
@parametrize_apply_noise_fn_name
def test_apply_noise_tensor_invariant(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    tensor_invariant_fn_name: TensorInvariantFnName,
    apply_noise_fn_name: ApplyNoiseFnName,
) -> None:
    """
    Tests that apply_noise maintains an invariant.

    Args:
        device_name: A device name
        float_dtype_name: A float dtype name
        tensor_invariant_fn_name: A tensor invariant name
        apply_noise_fn_name: A function name
    """
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    invariant_fn = get_tensor_invariant_fn(tensor_invariant_fn_name)
    apply_noise_fn = get_apply_noise_fn(apply_noise_fn_name)
    search_strategy = sized_3hw_tensors(device=device, dtype=dtype)

    @given(image=search_strategy, shot_noise=st.floats(0.0, 1.0), read_noise=st.floats(0.0, 1.0))
    def test(image: Tensor, shot_noise: float, read_noise: float) -> None:
        actual = apply_noise_fn(image, shot_noise, read_noise)
        invariant_holds = invariant_fn(image, actual)
        assert invariant_holds

    test()


@parametrize_device_name_float_dtype_name
@parametrize_apply_noise_fn_name
def test_apply_noise_has_identity_element(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    apply_noise_fn_name: ApplyNoiseFnName,
) -> None:
    """
    Tests that Noise(0.0, 0.0) is the identity under apply.

    Args:
        device_name: A device name
        float_dtype_name: A float dtype name
        apply_noise_fn_name: A function name
    """
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    apply_noise_fn = get_apply_noise_fn(apply_noise_fn_name)
    search_strategy = sized_3hw_tensors(device=device, dtype=dtype)

    @given(image=search_strategy)
    def test(image: Tensor) -> None:
        actual = apply_noise_fn(image, 0.0, 0.0)
        assert actual.allclose(image)

    test()
