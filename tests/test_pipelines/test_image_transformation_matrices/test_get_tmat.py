from typing import Callable

import cv2  # type: ignore[import]
import numpy as np
import numpy.typing as npt
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from mfsr_utils.pipelines.image_transformation_matrices import get_tmat as get_tmat


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
    t_rot = cv2.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)  # type: ignore
    t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))  # type: ignore

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


def given_get_tmat_args(f: Callable[..., None]):
    return given(
        image_shape=st.tuples(
            st.integers(min_value=32, max_value=1000),
            st.integers(min_value=32, max_value=1000),
        ),
        translation=st.tuples(
            st.floats(min_value=-8, max_value=8),
            st.floats(min_value=-8, max_value=8),
        ),
        theta=st.floats(min_value=-180.0, max_value=180.0),
        shear_values=st.tuples(
            st.floats(min_value=-1.0, max_value=1.0),
            st.floats(min_value=-1.0, max_value=1.0),
        ),
        scale_factors=st.tuples(
            st.floats(min_value=0.01, max_value=1.0),
            st.floats(min_value=0.01, max_value=1.0),
        ),
    )(f)


@given_get_tmat_args
def test_get_tmat_shape(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
) -> None:
    expected = (2, 3)
    actual = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    ).shape
    assert expected == actual


@given_get_tmat_args
def test_get_tmat_dtype(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
) -> None:
    expected = np.float64
    actual = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    ).dtype
    assert expected == actual


@pytest.mark.parametrize(
    "impl",
    [
        _get_tmat_reference,
        get_tmat,
    ],
)
@given_get_tmat_args
def test_get_tmat_shape_eq_impl_shape(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    impl: Callable[..., npt.NDArray[np.float64] | torch.Tensor],
) -> None:
    expected = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    ).shape
    actual = impl(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    ).shape
    assert expected == actual


@pytest.mark.parametrize(
    "impl",
    [
        _get_tmat_reference,
        get_tmat,
    ],
)
@given_get_tmat_args
def test_get_tmat_dtype_eq_impl_dtype(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    impl: Callable[..., npt.NDArray[np.float64] | torch.Tensor],
) -> None:
    expected_tmat = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    actual_tmat = impl(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )

    if isinstance(actual_tmat, torch.Tensor):
        expected = torch.from_numpy(expected_tmat).dtype  # type: ignore
    else:
        expected = expected_tmat.dtype

    actual = actual_tmat.dtype

    assert expected == actual


@pytest.mark.parametrize(
    "impl",
    [
        _get_tmat_reference,
        get_tmat,
    ],
)
@given_get_tmat_args
def test_get_tmat_values_eq_impl_values(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    impl: Callable[..., npt.NDArray[np.float64] | torch.Tensor],
) -> None:
    expected = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    actual = impl(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    if isinstance(actual, torch.Tensor):
        expected = torch.from_numpy(expected).to(  # type: ignore
            device=actual.device, dtype=actual.dtype
        )
        assert torch.allclose(expected, actual), f"{expected} != {actual}"
    else:
        assert np.allclose(expected, actual), f"{expected} != {actual}"  # type: ignore
