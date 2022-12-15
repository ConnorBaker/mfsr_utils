from typing import get_args

import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import Tensor

from mfsr_utils.pipelines.image_transformation_params import ImageTransformationParams
from mfsr_utils.pipelines.synthetic_burst_generator import single2lrburst
from mfsr_utils.pipelines.types import InterpolationType
from hypothesis_torch_utils.strategies._3hw_tensors import _3HW_TENSORS


@given(
    image=_3HW_TENSORS(),
    burst_size=st.just(2),
    # burst_size=st.integers(min_value=1, max_value=20),
    downsample_factor=st.just(1.0),
    # downsample_factor=st.floats(0.25, 1.0),
    transformation_params=st.none(),  # TODO: Add transformation params
    # interpolation_type=st.sampled_from(get_args(InterpolationType)),
    interpolation_type=st.just("bilinear"),
)
def test_single2lrburst_burst_size_matches(
    image: Tensor,
    burst_size: int,
    downsample_factor: float,
    transformation_params: None | ImageTransformationParams,
    interpolation_type: InterpolationType,
) -> None:
    burst_images, flow_vectors = single2lrburst(
        image,
        burst_size,
        downsample_factor,
        transformation_params,
        interpolation_type,
    )
    actual_burst_size = burst_images.shape[0]
    assert actual_burst_size == burst_size


@given(
    image=_3HW_TENSORS(),
    burst_size=st.just(2),
    # burst_size=st.integers(min_value=1, max_value=20),
    downsample_factor=st.just(1.0),
    # downsample_factor=st.floats(0.25, 1.0),
    transformation_params=st.none(),  # TODO: Add transformation params
    # interpolation_type=st.sampled_from(get_args(InterpolationType)),
    interpolation_type=st.just("bilinear"),
)
def test_single2lrburst_flow_shape_invariant(
    image: Tensor,
    burst_size: int,
    downsample_factor: float,
    transformation_params: None | ImageTransformationParams,
    interpolation_type: InterpolationType,
) -> None:
    burst_images, flow_vectors = single2lrburst(
        image,
        burst_size,
        downsample_factor,
        transformation_params,
        interpolation_type,
    )

    # The flow vectors have the same shape as the burst images but only have 2 channels
    expected_shape = burst_images.shape
    actual_shape = flow_vectors.shape

    assert actual_shape[0] == expected_shape[0]
    assert actual_shape[1] == 2
    assert actual_shape[2] == expected_shape[2]
    assert actual_shape[3] == expected_shape[3]


@given(
    image=_3HW_TENSORS(),
    burst_size=st.just(2),
    # burst_size=st.integers(min_value=1, max_value=20),
    downsample_factor=st.just(1.0),
    # downsample_factor=st.floats(0.25, 1.0),
    transformation_params=st.none(),  # TODO: Add transformation params
    # interpolation_type=st.sampled_from(get_args(InterpolationType)),
    interpolation_type=st.just("bilinear"),
)
def test_single2lrburst_device_invariant(
    image: Tensor,
    burst_size: int,
    downsample_factor: float,
    transformation_params: None | ImageTransformationParams,
    interpolation_type: InterpolationType,
) -> None:
    burst_images, flow_vectors = single2lrburst(
        image,
        burst_size,
        downsample_factor,
        transformation_params,
        interpolation_type,
    )

    expected_device = image.device
    actual_burst_device = burst_images.device
    actual_flow_device = flow_vectors.device
    assert actual_burst_device == expected_device
    assert actual_flow_device == expected_device


@given(
    image=_3HW_TENSORS(),
    burst_size=st.just(2),
    # burst_size=st.integers(min_value=1, max_value=20),
    downsample_factor=st.just(1.0),
    # downsample_factor=st.floats(0.25, 1.0),
    transformation_params=st.none(),  # TODO: Add transformation params
    # interpolation_type=st.sampled_from(get_args(InterpolationType)),
    interpolation_type=st.just("bilinear"),
)
def test_single2lrburst_dtype_invariant(
    image: Tensor,
    burst_size: int,
    downsample_factor: float,
    transformation_params: None | ImageTransformationParams,
    interpolation_type: InterpolationType,
) -> None:
    burst_images, flow_vectors = single2lrburst(
        image,
        burst_size,
        downsample_factor,
        transformation_params,
        interpolation_type,
    )

    expected_dtype = image.dtype
    actual_burst_dtype = burst_images.dtype
    actual_flow_dtype = flow_vectors.dtype
    assert actual_burst_dtype == expected_dtype
    assert actual_flow_dtype == expected_dtype
