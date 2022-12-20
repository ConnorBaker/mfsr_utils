from typing import Callable

import cv2  # type: ignore[import]
import numpy as np
import numpy.typing as npt
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis_torch_utils.strategies.devices_and_dtypes import DeviceAndDType, devices_and_dtypes
from hypothesis_torch_utils.strategies.dtypes import torch_float_dtypes
from torch import Tensor

from mfsr_utils.pipelines.image_transformation_matrices import get_tmat as get_tmat

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


def _get_tmat_reference(
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


def given_get_tmat_args(f: Callable[..., None]) -> Callable[..., None]:
    return given(
        image_shape=_IMAGE_SHAPES,
        translation=_TRANSLATIONS,
        theta=_THETAS,
        shear_values=_SHEAR_VALUES,
        scale_factors=_SCALE_FACTORS,
        device_and_dtype=devices_and_dtypes(dtype=torch_float_dtypes),
    )(f)


@given_get_tmat_args
def test_get_tmat_shape(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    device_and_dtype: DeviceAndDType,
) -> None:
    expected = (2, 3)
    actual = get_tmat(
        image_shape, translation, theta, shear_values, scale_factors, **device_and_dtype
    ).shape
    assert expected == actual


@given_get_tmat_args
def test_get_tmat_dtype(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    device_and_dtype: DeviceAndDType,
) -> None:
    expected = device_and_dtype["dtype"]
    actual = get_tmat(
        image_shape, translation, theta, shear_values, scale_factors, **device_and_dtype
    ).dtype
    assert expected == actual


@given_get_tmat_args
def test_get_tmat_device(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    device_and_dtype: DeviceAndDType,
) -> None:
    expected = device_and_dtype["device"]
    actual = get_tmat(
        image_shape, translation, theta, shear_values, scale_factors, **device_and_dtype
    ).device
    assert expected == actual


@given_get_tmat_args
def test_get_tmat_shape_eq_reference_impl_shape(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    device_and_dtype: DeviceAndDType,
) -> None:
    expected = _get_tmat_reference(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    ).shape
    actual = get_tmat(
        image_shape, translation, theta, shear_values, scale_factors, **device_and_dtype
    ).shape
    assert expected == actual


@given_get_tmat_args
def test_get_tmat_dtype_eq_reference_impl_dtype(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    device_and_dtype: DeviceAndDType,
) -> None:
    expected_tmat = _get_tmat_reference(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    # We don't use the dtype from device_and_dtype here because we're comparing matrices of
    # floating point values and cannot afford to lose precision.
    actual_tmat = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
        device=device_and_dtype["device"],
        dtype=torch.float64,
    )

    expected = torch.from_numpy(expected_tmat).dtype  # type: ignore
    actual = actual_tmat.dtype
    assert expected == actual


@given_get_tmat_args
def test_get_tmat_device_eq_reference_impl_device(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    device_and_dtype: DeviceAndDType,
) -> None:
    expected_tmat = _get_tmat_reference(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    # We don't use the device from device_and_dtype here because we're comparing the resulting
    # devices.
    actual_tmat = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
        device=torch.device("cpu"),
        dtype=device_and_dtype["dtype"],
    )

    expected = torch.from_numpy(expected_tmat).device  # type: ignore
    actual = actual_tmat.device
    assert expected == actual


@given_get_tmat_args
def test_get_tmat_values_eq_reference_impl_values(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    device_and_dtype: DeviceAndDType,
) -> None:
    _expected = _get_tmat_reference(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    # We don't use the dtype from device_and_dtype here because we're comparing matrices of
    # floating point values and cannot afford to lose precision.
    actual = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
        device=device_and_dtype["device"],
        dtype=torch.float64,
    )
    expected: Tensor = torch.from_numpy(_expected).to(  # type: ignore
        device=actual.device, dtype=actual.dtype
    )
    assert torch.allclose(expected, actual)
