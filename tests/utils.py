from functools import wraps
from typing import Any, Callable, Literal, ParamSpec, TypeAlias, TypeVar, cast, get_args

import pytest
import torch
from pytest import MarkDecorator
from torch import Tensor

P = ParamSpec("P")
R = TypeVar("R")

TensorInvariantFnTy: TypeAlias = Callable[[Tensor, Tensor], bool]
TensorInvariantFnName: TypeAlias = Literal["TensorShapeInvariant", "TensorDtypeInvariant", "TensorDeviceInvariant"]
parametrize_tensor_invariant_fn_name: MarkDecorator = pytest.mark.parametrize(
    "tensor_invariant_fn_name", get_args(TensorInvariantFnName)
)


def tensor_shape_invariant(x: Tensor, y: Tensor) -> bool:
    return x.shape == y.shape


def tensor_dtype_invariant(x: Tensor, y: Tensor) -> bool:
    return x.dtype == y.dtype


def tensor_device_invariant(x: Tensor, y: Tensor) -> bool:
    return x.device == y.device


def get_tensor_invariant_fn(
    tensor_invariant_fn_name: TensorInvariantFnName,
) -> TensorInvariantFnTy:
    match tensor_invariant_fn_name:
        case "TensorShapeInvariant":
            return tensor_shape_invariant
        case "TensorDtypeInvariant":
            return tensor_dtype_invariant
        case "TensorDeviceInvariant":
            return tensor_device_invariant


DeviceName: TypeAlias = Literal["cpu", "cuda:0"]


def parametrize_device_name(fn: Callable[P, R]) -> Callable[P, R]:
    @pytest.mark.parametrize("device_name", get_args(DeviceName))
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        _device_name: Any = kwargs.get("device_name")
        assert _device_name in get_args(DeviceName)
        device_name: DeviceName = cast(DeviceName, _device_name)

        device: torch.device = get_device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            pytest.xfail(reason="CUDA not available")

        try:
            return fn(*args, **kwargs)
        except AssertionError as e:
            str_e = str(e)
            if " not implemented " in str_e:
                pytest.xfail(reason=str_e)
            elif " not supported " in str_e:
                pytest.xfail(reason=str_e)
            else:
                raise e

    return wrapper


def get_device(device_name: DeviceName) -> torch.device:
    match device_name:
        case "cpu":
            return torch.device("cpu")
        case "cuda:0":
            return torch.device("cuda:0")


FloatDtypeName: TypeAlias = Literal["bfloat16", "float16", "float32", "float64"]
parametrize_float_dtype_name: MarkDecorator = pytest.mark.parametrize("float_dtype_name", get_args(FloatDtypeName))


def get_float_dtype(dtype_name: FloatDtypeName) -> torch.dtype:
    match dtype_name:
        case "bfloat16":
            return torch.bfloat16
        case "float16":
            return torch.float16
        case "float32":
            return torch.float32
        case "float64":
            return torch.float64


# Wrapper function which allows us to use the parametrize decorators
# and skips the test if the dtype is not supported by the device.
# Currently, bfloat16 isn't supported on CPU.
def parametrize_device_name_float_dtype_name(fn: Callable[P, R]) -> Callable[P, R]:
    @parametrize_device_name
    @parametrize_float_dtype_name
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        _device_name: Any = kwargs.get("device_name")
        assert _device_name in get_args(DeviceName)
        device_name: DeviceName = cast(DeviceName, _device_name)

        device: torch.device = get_device(device_name)

        _float_dtype_name: Any = kwargs.get("float_dtype_name")
        assert _float_dtype_name in get_args(FloatDtypeName)
        float_dtype_name: FloatDtypeName = cast(FloatDtypeName, _float_dtype_name)

        dtype: torch.dtype = get_float_dtype(float_dtype_name)
        if dtype == torch.float16 and device.type == "cpu":
            pytest.xfail(reason="float16 not supported on CPU")

        try:
            return fn(*args, **kwargs)
        except AssertionError as e:
            str_e = str(e)
            if " not implemented " in str_e:
                pytest.xfail(reason=str_e)
            elif " not supported " in str_e:
                pytest.xfail(reason=str_e)
            else:
                raise e

    return wrapper
