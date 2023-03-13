from pathlib import Path
from dataclasses import dataclass
from typing import Any, Literal, Type, get_args
from typing_extensions import Self
from numbers import Integral
import torch
import numpy as np
from numpy import typing as npt
from torch import Tensor
import rawpy

RawType = Literal["Flat", "Stack"]
ColorDesc = Literal["RGBG", "RGBE", "GMCY", "GBTG"]


@dataclass
class ImageSizes:
    raw_height: int
    raw_width: int
    height: int
    width: int
    top_margin: int
    left_margin: int
    iheight: int
    iwidth: int
    pixel_aspect: float
    flip: int

    @classmethod
    def from_rawpy(cls: Type[Self], image_sizes: Any) -> Self:
        return cls(**image_sizes._asdict())  # type: ignore


@dataclass
class RawImage:
    black_level_per_channel: list[int]
    camera_white_level_per_channel: list[int]
    camera_whitebalance: list[float]
    color_desc: ColorDesc
    color_matrix: npt.NDArray[np.float32]
    daylight_whitebalance: list[float]
    num_colors: int
    raw_colors: npt.NDArray[np.uint8]
    raw_colors_visible: npt.NDArray[np.uint8]
    raw_image: npt.NDArray[np.uint16]
    raw_image_visible: npt.NDArray[np.uint16]
    # raw_pattern gives us the indices we need to permute to get to RGGB
    # For example, raw_pattern = [[0, 1], [3, 2]] means we have RGBG
    raw_pattern: npt.NDArray[np.uint8]
    raw_type: RawType
    rgb_xyz_matrix: npt.NDArray[np.float32]
    sizes: ImageSizes
    tone_curve: npt.NDArray[np.uint16]
    white_level: int

    @classmethod
    def from_file(cls: Type[Self], filename: str | Path) -> Self:
        with rawpy.imread(filename) as raw:  # type: ignore
            return cls.from_rawpy(raw)

    @classmethod
    def from_rawpy(cls: Type[Self], raw: Any) -> Self:
        d: dict[str, Any] = {k: getattr(raw, k) for k in cls.__dataclass_fields__.keys()}
        d["color_desc"] = d["color_desc"].decode("utf-8")
        assert d["color_desc"] in get_args(ColorDesc)
        if d["raw_type"].value == 0:
            d["raw_type"] = "Flat"
        elif d["raw_type"].value == 1:
            d["raw_type"] = "Stack"
        else:
            raise ValueError(f"Unknown raw_type {d['raw_type']}")
        d["sizes"] = ImageSizes.from_rawpy(d["sizes"])
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.copy()
        return cls(**d)

    def to_tensor(
        self, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        # Don't want to figure out how to scale floating point images, so only accept RAW files
        # which store integers.
        raw_image = self.raw_image_visible
        assert issubclass(raw_image.dtype.type, Integral)
        assert raw_image.min() > 0  # type: ignore

        # Get the maximum value for the dtype before we potentially cast to a larger dtype
        dtype_max = np.iinfo(raw_image.dtype).max

        # Can't convert np.uint16 directly to a torch tensor, so convert to np.int32 first
        if raw_image.dtype == np.uint16:
            raw_image = raw_image.astype(np.int32)

        # Create a tensor from the raw image, cast to float64, and normalize to [0, 1], then
        # cast to the desired dtype.
        tensor: Tensor = (
            torch.from_numpy(raw_image)  # type: ignore
            .to(dtype=torch.float64, device=device)
            .div(dtype_max)
            .to(dtype=dtype, device=device)
        )
        return tensor
