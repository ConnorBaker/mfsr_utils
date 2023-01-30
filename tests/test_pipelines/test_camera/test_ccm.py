from typing import Callable, Literal, cast, get_args

import pytest
import torch
from hypothesis import given
from hypothesis_torch_utils.strategies.sized_3hw_tensors import sized_3hw_tensors
from torch import Tensor

from mfsr_utils.pipelines.camera import apply_ccm, random_ccm
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

ApplyCCMFnTy = Callable[[Tensor, Tensor], Tensor]
ApplyCCMFnName = Literal["apply_ccm", "compiled_apply_ccm"]
compiled_apply_ccm = cast(ApplyCCMFnTy, torch.compile(apply_ccm))  # type: ignore
parametrize_apply_ccm_fn_name = pytest.mark.parametrize(
    "apply_ccm_fn_name", get_args(ApplyCCMFnName)
)


def get_apply_ccm_fn(apply_ccm_fn_name: ApplyCCMFnName) -> ApplyCCMFnTy:
    match apply_ccm_fn_name:
        case "apply_ccm":
            return apply_ccm
        case "compiled_apply_ccm":
            return compiled_apply_ccm


@parametrize_device_name_float_dtype_name
@parametrize_tensor_invariant_fn_name
def test_random_ccm_tensor_invariant(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    tensor_invariant_fn_name: TensorInvariantFnName,
) -> None:
    """
    Tests that gamma_fn maintains an invariant.

    Args:
        device_name: A device name
        float_dtype_name: A float dtype name
        tensor_invariant_fn_name: A tensor invariant name
    """
    device = get_device(device_name)
    dtype = get_float_dtype(float_dtype_name)

    expected = torch.empty((3, 3), dtype=dtype, device=device)
    actual = random_ccm(dtype=dtype, device=device)

    invariant_fn = get_tensor_invariant_fn(tensor_invariant_fn_name)
    invariant_holds = invariant_fn(expected, actual)
    assert invariant_holds


@parametrize_device_name_float_dtype_name
def test_random_ccm_rows_sum_to_1(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
) -> None:
    """
    Tests that the CCM from random_ccm() has rows which sum to 1.

    Args:
        device_name: A device name
        float_dtype_name: A float dtype name
    """
    device = get_device(device_name)
    dtype = get_float_dtype(float_dtype_name)

    expected = torch.ones(3, dtype=dtype, device=device)
    actual = random_ccm(dtype=dtype, device=device).sum(dim=1)
    match dtype:
        case torch.bfloat16:
            rtol = 1e-02
        case torch.float16:
            rtol = 1e-03
        case torch.float32:
            rtol = 1e-04
        case _:
            rtol = 1e-05

    rows_sum_to_1 = actual.allclose(expected, rtol=rtol)
    assert rows_sum_to_1


@parametrize_device_name_float_dtype_name
@parametrize_tensor_invariant_fn_name
@parametrize_apply_ccm_fn_name
def test_apply_ccm_tensor_invariant(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    tensor_invariant_fn_name: TensorInvariantFnName,
    apply_ccm_fn_name: ApplyCCMFnName,
) -> None:
    """
    Tests that gamma_fn maintains an invariant.

    Args:
        device_name: A device name
        float_dtype_name: A float dtype name
        tensor_invariant_fn_name: A tensor invariant name
        apply_ccm_fn_name: A function name
    """
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    invariant_fn = get_tensor_invariant_fn(tensor_invariant_fn_name)
    apply_ccm_fn = get_apply_ccm_fn(apply_ccm_fn_name)
    search_strategy = sized_3hw_tensors(device=device, dtype=dtype)

    @given(image=search_strategy)
    def test(image: Tensor) -> None:
        ccm = random_ccm(dtype=image.dtype, device=image.device)
        actual = apply_ccm_fn(image, ccm)
        invariant_holds = invariant_fn(image, actual)
        assert invariant_holds

    test()
