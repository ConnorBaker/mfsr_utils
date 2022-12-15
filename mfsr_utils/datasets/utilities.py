import numpy as np
import torch
from numpy import floating, ndarray
from numpy.typing import NBitBase, NDArray
from torch import Tensor
from typing_extensions import TypeVar, overload

_T = TypeVar("_T", bound=NBitBase)


@overload
def pack_raw_image(im_raw: NDArray[floating[_T]]) -> NDArray[floating[_T]]:
    ...


@overload
def pack_raw_image(im_raw: Tensor) -> Tensor:
    ...


def pack_raw_image(
    im_raw: NDArray[floating[_T]] | Tensor,
) -> NDArray[floating[_T]] | Tensor:
    im_out: NDArray[floating[_T]] | Tensor
    new_shape = (4, im_raw.shape[0] // 2, im_raw.shape[1] // 2)
    match im_raw:
        case ndarray():
            im_out = np.zeros_like(im_raw).reshape(new_shape)
        case Tensor():
            im_out = torch.zeros_like(im_raw).reshape(new_shape)

    # Manually unroll the assignment loop because it's faster.
    # Notice that we're effectively counting in binary on the right hand side.
    im_out[0, :, :] = im_raw[0::2, 0::2]  # type: ignore
    im_out[1, :, :] = im_raw[0::2, 1::2]  # type: ignore
    im_out[2, :, :] = im_raw[1::2, 0::2]  # type: ignore
    im_out[3, :, :] = im_raw[1::2, 1::2]  # type: ignore

    return im_out


@overload
def flatten_raw_image(im_raw_4ch: NDArray[floating[_T]]) -> NDArray[floating[_T]]:
    ...


@overload
def flatten_raw_image(im_raw_4ch: Tensor) -> Tensor:
    ...


def flatten_raw_image(
    im_raw_4ch: NDArray[floating[_T]] | Tensor,
) -> NDArray[floating[_T]] | Tensor:
    im_out: NDArray[floating[_T]] | Tensor
    new_shape = (3, im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2)
    match im_raw_4ch:
        case ndarray():
            im_out = np.zeros_like(im_raw_4ch).reshape(new_shape)
        case Tensor():
            im_out = torch.zeros_like(im_raw_4ch).reshape(new_shape)

    # Manually unroll the assignment loop because it's faster.
    # Notice that we're effectively counting in binary on the left hand side.
    im_out[0::2, 0::2] = im_raw_4ch[0, :, :]  # type: ignore
    im_out[0::2, 1::2] = im_raw_4ch[1, :, :]  # type: ignore
    im_out[1::2, 0::2] = im_raw_4ch[2, :, :]  # type: ignore
    im_out[1::2, 1::2] = im_raw_4ch[3, :, :]  # type: ignore

    return im_out
