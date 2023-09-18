from typing import Callable, Literal, get_args

import pytest
import torch
from hypothesis import given
from hypothesis_torch_utils.strategies.sized_3hw_tensors import sized_3hw_tensors
from torch import Tensor

from mfsr_utils.pipelines.camera import gamma_compression, gamma_expansion
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

GammaFnTy = Callable[[Tensor], Tensor]
GammaExpansionFnName = Literal["gamma_expansion"]
parametrize_gamma_expansion_fn_name = pytest.mark.parametrize("gamma_expansion_fn_name", get_args(GammaExpansionFnName))
GammaCompressionFnName = Literal["gamma_compression"]
parametrize_gamma_compression_fn_name = pytest.mark.parametrize(
    "gamma_compression_fn_name", get_args(GammaCompressionFnName)
)
GammaFnName = Literal[GammaExpansionFnName, GammaCompressionFnName]
parametrize_gamma_fn_name = pytest.mark.parametrize("gamma_fn_name", get_args(GammaFnName))


def get_gamma_fn(gamma_fn_name: GammaFnName) -> GammaFnTy:
    match gamma_fn_name:
        case "gamma_expansion":
            return gamma_expansion
        case "gamma_compression":
            return gamma_compression


GammaComposedFnName = Literal["expansion_then_compression", "compression_then_expansion"]
parametrize_gamma_composed_fn_name = pytest.mark.parametrize("gamma_composed_fn_name", get_args(GammaComposedFnName))


def get_gamma_composed_fn(
    gamma_composed_fn_name: GammaComposedFnName,
    gamma_expansion_fn_name: GammaExpansionFnName,
    gamma_compression_fn_name: GammaCompressionFnName,
) -> GammaFnTy:
    gamma_expansion_fn = get_gamma_fn(gamma_expansion_fn_name)
    gamma_compression_fn = get_gamma_fn(gamma_compression_fn_name)
    match gamma_composed_fn_name:
        case "expansion_then_compression":
            return lambda x: gamma_compression_fn(gamma_expansion_fn(x))
        case "compression_then_expansion":
            return lambda x: gamma_expansion_fn(gamma_compression_fn(x))


@parametrize_device_name_float_dtype_name
@parametrize_tensor_invariant_fn_name
@parametrize_gamma_fn_name
def test_gamma_fn_tensor_invariant(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    tensor_invariant_fn_name: TensorInvariantFnName,
    gamma_fn_name: GammaFnName,
) -> None:
    """
    Tests that gamma_fn maintains an invariant.

    Args:
        device_name: The name of the device to run the test on
        float_dtype_name: The name of the floating dtype to use
        tensor_invariant_fn_name: The name of the invariant to test
        gamma_fn_name: The name of the gamma function to test
    """
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    invariant_fn = get_tensor_invariant_fn(tensor_invariant_fn_name)
    gamma_fn = get_gamma_fn(gamma_fn_name)
    search_strategy = sized_3hw_tensors(device=device, dtype=dtype)

    @given(image=search_strategy)
    def test(image: Tensor) -> None:
        actual = gamma_fn(image)
        invariant_holds = invariant_fn(image, actual)
        assert invariant_holds

    test()


@parametrize_device_name_float_dtype_name
@parametrize_gamma_composed_fn_name
@parametrize_gamma_expansion_fn_name
@parametrize_gamma_compression_fn_name
def test_gamma_fn_inverse(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    gamma_composed_fn_name: GammaComposedFnName,
    gamma_expansion_fn_name: GammaExpansionFnName,
    gamma_compression_fn_name: GammaCompressionFnName,
) -> None:
    """
    Tests that gamma_expansion is the inverse of gamma_compression (roughly).

    Args:
        device_name: The name of the device to run the test on
        float_dtype_name: The name of the floating dtype to use
        gamma_composed_fn_name: The name of the composed gamma function to test
        gamma_expansion_fn_name: The name of the gamma expansion function to test
        gamma_compression_fn_name: The name of the gamma compression function to test
    """
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    gamma_composed_fn = get_gamma_composed_fn(
        gamma_composed_fn_name, gamma_expansion_fn_name, gamma_compression_fn_name
    )

    search_strategy = sized_3hw_tensors(device=device, dtype=dtype)

    @given(image=search_strategy)
    def test(image: Tensor) -> None:
        actual = gamma_composed_fn(image)
        assert actual.allclose(image, rtol=1e-2, atol=1e-3)

    test()
