from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Type, TypeVar, get_args

import numpy as np
import rawpy
import torch
from torch import Tensor, nn
from torch.utils.data.dataset import Dataset
from typing_extensions import Self

_T = TypeVar("_T")

# Truly, RawType can be either "Flat" or "Stack", but for the purposes of this dataset, we consider
# only "Flat" images.
RawType = Literal["Flat"]

# Truly, ColorDesc can be one of "RGBG", "RGBE", "GMCY", or "GBTG", but for the purposes of this
# dataset, we consider only "RGBG" images.
ColorDesc = Literal["RGBG"]


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
class ArwImage:
    """Sony ARW image."""

    path: Path
    black_level_per_channel: list[int]
    camera_white_level_per_channel: list[int]
    camera_whitebalance: list[float]
    color_desc: ColorDesc
    color_matrix: Tensor
    daylight_whitebalance: list[float]
    # num_colors: int
    # raw_colors: npt.NDArray[np.uint8]
    # raw_colors_visible: npt.NDArray[np.uint8]
    # raw_image: npt.NDArray[np.uint16]
    raw_image_visible: Tensor
    # raw_pattern: npt.NDArray[np.uint8]
    raw_type: RawType
    rgb_xyz_matrix: Tensor
    sizes: ImageSizes
    tone_curve: Tensor
    white_level: int

    @classmethod
    def from_path(cls: Type[Self], path: Path) -> Self:
        """
        Creates an ArwImage from a file.

        Args:
            filename (Path): The path to the file.

        Returns:
            An ArwImage object.
        """
        with rawpy.imread(path.as_posix()) as raw:  # type: ignore
            d: dict[str, Any] = {
                k: getattr(raw, k) if k != "path" else path
                for k in cls.__dataclass_fields__.keys()
            }

            # Convert color_desc to a string
            d["color_desc"] = d["color_desc"].decode("utf-8")
            assert d["color_desc"] in get_args(
                ColorDesc
            ), f"Unsupported color_desc {d['color_desc']}. "

            # Convert raw_type to a string
            assert d["raw_type"].value == 0, f"Unsupported raw_type {d['raw_type']}"
            d["raw_type"] = "Flat"

            d["sizes"] = ImageSizes.from_rawpy(d["sizes"])

            # Convert numpy arrays to torch tensors
            d["color_matrix"] = torch.from_numpy(d["color_matrix"].copy())
            d["raw_image_visible"] = torch.from_numpy(
                # TODO: This copy may not be necessary because we do div afterwards, which creates
                # a new tensor.
                d["raw_image_visible"]
                .astype(np.float32)
                .copy()
            ).div(np.iinfo(d["raw_image_visible"].dtype).max)
            d["rgb_xyz_matrix"] = torch.from_numpy(d["rgb_xyz_matrix"].copy())
            d["tone_curve"] = torch.from_numpy(d["tone_curve"].astype(np.int32).copy())

            return cls(**d)


@dataclass
class SonyArw(Dataset[_T]):
    """
    A dataset of Sony ARW images. Reads the images from the directory `data_dir` and returns
    them as tensors after applying the transform `transform`.

    Args:
        data_dir (Path): Path to the directory containing the ARW images.
        transform (Callable[[ArwImage], _T]): A function/transform that takes in an ArwImage
            and returns a transformed version. E.g, `transforms.ToTensor()`.

    Returns:
        A dataset of Sony ARW images.

    See Also:
        `mosaic_rgbg` in `mfsr_utils.pipelines.camera`
    """

    data_dir: Path
    transform: Callable[[ArwImage], _T] = nn.Identity()
    files: list[Path] = field(init=False)

    def __post_init__(self) -> None:
        self.files = list(self.data_dir.glob("*.ARW"))

    def __getitem__(self, index: int) -> _T:
        image_path = self.files[index]
        raw_image = ArwImage.from_path(image_path)
        transformed: _T = self.transform(raw_image)
        return transformed

    def __len__(self) -> int:
        return len(self.files)
