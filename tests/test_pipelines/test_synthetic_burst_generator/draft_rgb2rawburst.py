import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies._internal.core import RandomSeeder
from hypothesis_torch_utils.strategies.sized_3hw_tensors import sized_3hw_tensors
from torch import Tensor
from typing_extensions import get_args

from mfsr_utils.pipelines.image_processing_params import ImageProcessingParams
from mfsr_utils.pipelines.image_transform_params import ImageTransformParams
from mfsr_utils.pipelines.types import InterpolationType


@given(
    rs=st.random_module(),
    image=sized_3hw_tensors(dtype=torch.float32),
    burst_size=st.integers(min_value=1, max_value=20),
    downsample_factor=st.floats(0.25, 1.0),
    burst_transform_params=st.none(),  # TODO: Add transformation params
    image_processing_params=st.none(),  # TODO: Add image processing params
    interpolation_type=st.sampled_from(get_args(InterpolationType)),
)
def test_rgb2rawburst(
    rs: RandomSeeder,
    image: Tensor,
    burst_size: int,
    downsample_factor: float,
    burst_transform_params: None | ImageTransformParams,
    image_processing_params: None | ImageProcessingParams,
    interpolation_type: InterpolationType,
) -> None:
    """
    Tests converting an RGB image to a raw burst.

    Ensures:
        - image_burst is 4D
        - The length of the burst dimension is burst_size
        - image_burst has the same device as the input image
        - image_burst has the same dtype as the input image
        - TODO: What else?

    Args:
        rs: Hypothesis random module
        image: Image to convert to raw burst
        burst_size: Number of images in the burst
        downsample_factor: Factor to downsample the image by
        burst_transform_params: Transformation parameters to apply to the burst
        image_processing_params: Image processing parameters to apply to the burst
        interpolation_type: Interpolation type to use when resizing the image

    Returns:
        Tuple of:
            - Mosaiced raw burst with post-processing applied
            - Image with post-processing applied
            - Raw burst with post-processing applied
            - Flow vectors
            - Meta info
    """
    assert False, "TODO: Finish writing test"
