from typing import Callable, Literal, get_args

import cv2  # type: ignore[import]
import numpy as np
import numpy.typing as npt
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import Tensor

from mfsr_utils.pipelines.image_transform_mats import get_transform_mat as get_transform_mat
from tests.utils import (
    DeviceName,
    FloatDtypeName,
    TensorInvariantFnName,
    get_device,
    get_float_dtype,
    get_tensor_invariant_fn,
    parametrize_device_name,
    parametrize_device_name_float_dtype_name,
    parametrize_tensor_invariant_fn_name,
)

GetTransformMatFnTy = Callable[
    [
        tuple[int, int],
        tuple[float, float],
        float,
        tuple[float, float],
        tuple[float, float],
        torch.dtype,
        torch.device,
    ],
    Tensor,
]
GetTransformMatFnName = Literal["get_transform_mat"]
parametrize_get_transform_mat_fn_name = pytest.mark.parametrize(
    "get_transform_mat_fn_name", get_args(GetTransformMatFnName)
)


def get_transform_mat_fn(get_transform_mat_fn_name: GetTransformMatFnName) -> GetTransformMatFnTy:
    match get_transform_mat_fn_name:
        case "get_transform_mat":
            return get_transform_mat


_IMAGE_SHAPES: st.SearchStrategy[tuple[int, int]] = st.tuples(
    st.integers(min_value=32, max_value=1000),
    st.integers(min_value=32, max_value=1000),
)

_TRANSLATIONS: st.SearchStrategy[tuple[float, float]] = st.tuples(
    st.floats(min_value=-8, max_value=8),
    st.floats(min_value=-8, max_value=8),
)

_THETAS: st.SearchStrategy[float] = st.floats(min_value=-180.0, max_value=180.0)

_SHEAR_VALUES: st.SearchStrategy[tuple[float, float]] = st.tuples(
    st.floats(min_value=-1.0, max_value=1.0),
    st.floats(min_value=-1.0, max_value=1.0),
)

_SCALE_FACTORS: st.SearchStrategy[tuple[float, float]] = st.tuples(
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
)


def _get_transform_mat_reference(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
) -> npt.NDArray[np.float64]:
    """Generates a transformation matrix corresponding to the input transformation parameters"""
    im_h, im_w = image_shape

    t_mat = np.identity(3)

    t_mat[0, 2] = translation[0]
    t_mat[1, 2] = translation[1]
    # The type ignores which follow are for Pyright
    t_rot: npt.NDArray[np.float64] = cv2.getRotationMatrix2D(  # type: ignore[attr-defined]
        (im_w * 0.5, im_h * 0.5), theta, 1.0
    )
    assert t_rot.dtype == np.float64
    t_rot = np.concatenate(  # type: ignore[attr-defined]
        (t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3))
    )
    assert t_rot.dtype == np.float64

    t_shear = np.array(
        [
            [1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
            [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
            [0.0, 0.0, 1.0],
        ]
    )

    t_scale = np.array(
        [[scale_factors[0], 0.0, 0.0], [0.0, scale_factors[1], 0.0], [0.0, 0.0, 1.0]]
    )

    t_mat = t_scale @ t_rot @ t_shear @ t_mat

    t_mat = t_mat[:2, :]

    return t_mat


given_get_transform_mat_args = given(
    image_shape=_IMAGE_SHAPES,
    translation=_TRANSLATIONS,
    theta=_THETAS,
    shear_values=_SHEAR_VALUES,
    scale_factors=_SCALE_FACTORS,
)


@parametrize_device_name_float_dtype_name
@parametrize_tensor_invariant_fn_name
@parametrize_get_transform_mat_fn_name
@given_get_transform_mat_args
def test_get_transform_mat_tensor_invariant(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    tensor_invariant_fn_name: TensorInvariantFnName,
    get_transform_mat_fn_name: GetTransformMatFnName,
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
) -> None:
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    invariant_fn = get_tensor_invariant_fn(tensor_invariant_fn_name)
    transform_mat_fn = get_transform_mat_fn(get_transform_mat_fn_name)

    expected = torch.empty((2, 3), device=device, dtype=dtype)

    actual = transform_mat_fn(
        image_shape, translation, theta, shear_values, scale_factors, dtype, device
    )
    invariant_holds = invariant_fn(expected, actual)
    assert invariant_holds


@parametrize_device_name_float_dtype_name
@parametrize_get_transform_mat_fn_name
@given_get_transform_mat_args
def test_get_transform_mat_shape_eq_reference_impl_shape(
    device_name: DeviceName,
    float_dtype_name: FloatDtypeName,
    get_transform_mat_fn_name: GetTransformMatFnName,
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
) -> None:
    device: torch.device = get_device(device_name)
    dtype: torch.dtype = get_float_dtype(float_dtype_name)
    transform_mat_fn = get_transform_mat_fn(get_transform_mat_fn_name)

    expected = _get_transform_mat_reference(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    ).shape
    actual = transform_mat_fn(
        image_shape, translation, theta, shear_values, scale_factors, dtype, device
    ).shape
    assert expected == actual


@parametrize_device_name
@parametrize_get_transform_mat_fn_name
@given_get_transform_mat_args
def test_get_transform_mat_values_eq_reference_impl_values(
    device_name: DeviceName,
    get_transform_mat_fn_name: GetTransformMatFnName,
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
) -> None:
    device: torch.device = get_device(device_name)
    transform_mat_fn = get_transform_mat_fn(get_transform_mat_fn_name)

    _expected = _get_transform_mat_reference(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    # We don't use the dtype from device_and_dtype here because we're comparing matrices of
    # floating point values and cannot afford to lose precision.
    actual = transform_mat_fn(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
        torch.float64,
        device,
    )
    expected: Tensor = torch.from_numpy(_expected).to(  # type: ignore
        device=actual.device, dtype=actual.dtype
    )
    assert actual.allclose(expected)
