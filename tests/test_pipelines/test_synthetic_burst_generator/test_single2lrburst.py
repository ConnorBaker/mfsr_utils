import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis_torch_utils.strategies.sized_3hw_tensors import sized_3hw_tensors
from torch import Tensor

from mfsr_utils.pipelines.image_transform_params import ImageTransformParams
from mfsr_utils.pipelines.synthetic_burst_generator import single2lrburst
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


@parametrize_device_name_float_dtype_name
@parametrize_tensor_invariant_fn_name
def test_single2lrburst_tensor_invariant(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    tensor_invariant_fn_name: TensorInvariantFnName,
) -> None:
    """
    Tests that single2lrburst maintains an invariant.

    Args:
        device_name: A device name
        float_dtype_name: A float dtype name
        tensor_invariant_fn_name: A tensor invariant name
    """
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    invariant_fn = get_tensor_invariant_fn(tensor_invariant_fn_name)
    search_strategy = sized_3hw_tensors(device=device, dtype=dtype)

    @given(
        image=search_strategy,
        burst_size=st.integers(min_value=1, max_value=20),
        downsample_factor=st.floats(1.0, 4.0),
        transform_params=st.none(),  # TODO: Add transformation params
    )
    def test(
        image: Tensor,
        burst_size: int,
        downsample_factor: float,
        transform_params: None | ImageTransformParams,
    ) -> None:
        try:
            actual = single2lrburst(
                image,
                burst_size,
                downsample_factor,
                transform_params,
            )
            expected = image
            if tensor_invariant_fn_name == "TensorShapeInvariant":
                C, H, W = image.shape
                expected = torch.empty(
                    size=(burst_size, C, int(H / downsample_factor), int(W / downsample_factor)),
                    dtype=dtype,
                    device=device,
                )

            invariant_holds = invariant_fn(expected, actual)
            assert invariant_holds, f"Expected: {expected.shape}, Actual: {actual.shape}"
        except RuntimeError as e:
            if (
                device.type == "cpu"
                and dtype == torch.bfloat16
                and "grid_sampler_2d_cpu not implemented for BFloat16" in str(e)
            ):
                pytest.xfail("grid_sampler_2d_cpu not implemented for BFloat16")

    test()
