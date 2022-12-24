from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torchvision  # type: ignore[import]
from torch import Tensor
from torch.nn import Identity
from torch.utils.data.dataset import Dataset
from typing_extensions import ClassVar, TypeVar

from mfsr_utils.datasets.protocols.downloadable import Downloadable

_T = TypeVar("_T", default=Tensor)


@dataclass
class ZurichRaw2Rgb(Dataset[_T], Downloadable):  # type: ignore[valid-type]
    """Canon RGB images from the "Zurich RAW to RGB mapping" dataset. You can download the full
    dataset (22 GB) from http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset. Alternatively, you
    can only download the Canon RGB images (5.5 GB) from
    https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip
    """

    url: ClassVar[str] = "https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip"
    filename: ClassVar[str] = "zurich-raw-to-rgb.zip"
    dirname: ClassVar[str] = "zurich-raw-to-rgb"
    mirrors: ClassVar[list[str]] = [
        "https://storage.googleapis.com/bsrt-supplemental/zurich-raw-to-rgb.zip"
    ]

    data_dir: Path
    transform: Callable[[Tensor], _T] = field(default_factory=Identity)  # type: ignore[valid-type]

    def __getitem__(self, index: int) -> _T:  # type: ignore[valid-type]
        image_path = self.data_dir / self.dirname / "train" / "canon" / f"{index}.jpg"
        image_file: Tensor = torchvision.io.read_file(image_path.as_posix())
        image_jpg: Tensor = torchvision.io.decode_jpeg(image_file)
        transformed: _T = self.transform(image_jpg)  # type: ignore[valid-type]
        return transformed

    def __len__(self) -> int:
        return 46839
