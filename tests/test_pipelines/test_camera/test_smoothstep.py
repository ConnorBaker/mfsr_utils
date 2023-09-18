from typing import Callable, Literal, get_args

import pytest
import torch
from hypothesis import given
from hypothesis_torch_utils.strategies.sized_3hw_tensors import sized_3hw_tensors
from torch import Tensor
from torch.nn import functional as F

from mfsr_utils.pipelines.camera import apply_smoothstep, invert_smoothstep
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

SmoothstepFnTy = Callable[[Tensor], Tensor]
ApplySmoothstepFnName = Literal["apply_smoothstep"]
parametrize_apply_smoothstep_fn_name = pytest.mark.parametrize(
    "apply_smoothstep_fn_name", get_args(ApplySmoothstepFnName)
)
InvertSmoothstepFnName = Literal["invert_smoothstep"]
parametrize_invert_smoothstep_fn_name = pytest.mark.parametrize(
    "invert_smoothstep_fn_name", get_args(InvertSmoothstepFnName)
)
SmoothstepFnName = Literal[ApplySmoothstepFnName, InvertSmoothstepFnName]
parametrize_smoothstep_fn_name = pytest.mark.parametrize("smoothstep_fn_name", get_args(SmoothstepFnName))


def get_smoothstep_fn(smoothstep_fn_name: SmoothstepFnName) -> SmoothstepFnTy:
    match smoothstep_fn_name:
        case "apply_smoothstep":
            return apply_smoothstep
        case "invert_smoothstep":
            return invert_smoothstep


SmoothstepComposedFnName = Literal["apply_then_invert", "invert_then_apply"]
parametrize_smoothstep_composed_fn_name = pytest.mark.parametrize(
    "smoothstep_composed_fn_name", get_args(SmoothstepComposedFnName)
)


@parametrize_device_name_float_dtype_name
@parametrize_tensor_invariant_fn_name
@parametrize_smoothstep_fn_name
def test_smoothstep_fn_tensor_invariant(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    tensor_invariant_fn_name: TensorInvariantFnName,
    smoothstep_fn_name: SmoothstepFnName,
) -> None:
    """
    Tests that smoothstep_fn maintains an invariant.

    Args:
        image: A 3HW tensor of floating dtype
        tensor_invariant_fn_name: The name of the invariant to test
        smoothstep_fn_name: The name of the smoothstep function
    """

    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    invariant_fn = get_tensor_invariant_fn(tensor_invariant_fn_name)
    smoothstep_fn = get_smoothstep_fn(smoothstep_fn_name)
    search_strategy = sized_3hw_tensors(device=device, dtype=dtype)

    @given(image=search_strategy)
    def test(image: Tensor) -> None:
        actual = smoothstep_fn(image)
        invariant_holds = invariant_fn(image, actual)
        assert invariant_holds

    test()


@parametrize_device_name_float_dtype_name
@parametrize_apply_smoothstep_fn_name
@parametrize_invert_smoothstep_fn_name
@parametrize_smoothstep_composed_fn_name
def test_smoothstep_fn_inverse(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    apply_smoothstep_fn_name: ApplySmoothstepFnName,
    invert_smoothstep_fn_name: InvertSmoothstepFnName,
    smoothstep_composed_fn_name: SmoothstepComposedFnName,
) -> None:
    """
    Tests that applying the smoothstep function and then inverting it is the identity.

    Args:
        device_name: The name of the device to run the test on
        float_dtype_name: The name of the floating dtype to use
    """
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    apply_smoothstep_fn = get_smoothstep_fn(apply_smoothstep_fn_name)
    invert_smoothstep_fn = get_smoothstep_fn(invert_smoothstep_fn_name)
    search_strategy = sized_3hw_tensors(device=device, dtype=dtype)

    @given(image=search_strategy)
    def test(image: Tensor) -> None:
        match smoothstep_composed_fn_name:
            case "apply_then_invert":
                pre = apply_smoothstep_fn(image)
                post = invert_smoothstep_fn(pre)
            case "invert_then_apply":
                pre = invert_smoothstep_fn(image)
                post = apply_smoothstep_fn(pre)

        if image.device.type == "cpu" and image.dtype == torch.bfloat16:
            # MSE loss is not supported on CPU for bfloat16
            image = image.to(torch.float32)
            post = post.to(torch.float32)
            pre = pre.to(torch.float32)

        # The smoothstep function is not invertible, so we can only test that the result is close
        # to the original image.
        pre_mse = F.mse_loss(image, pre)
        post_mse = F.mse_loss(image, post)
        assert post_mse < pre_mse
        assert pre_mse <= 0.005

    test()
