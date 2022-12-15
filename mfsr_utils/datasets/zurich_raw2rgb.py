from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torchvision  # type: ignore[import]
from torch import Tensor
from torchvision.datasets import VisionDataset  # type: ignore[import]
from typing_extensions import ClassVar

from mfsr_utils.datasets.utilities.downloadable import DownloadableMixin


@dataclass
class ZurichRaw2Rgb(VisionDataset, DownloadableMixin):
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

    data_dir: str
    transform: None | Callable[[Tensor], Tensor | dict[str, Tensor]] = None

    def __getitem__(self, index: int) -> Tensor | dict[str, Tensor]:
        image_path = Path(self.data_dir) / self.dirname / "train" / "canon" / f"{index}.jpg"
        image_file: Tensor = torchvision.io.read_file(image_path.as_posix())
        image_jpg: Tensor = torchvision.io.decode_jpeg(image_file)
        transformed = image_jpg if self.transform is None else self.transform(image_jpg)
        return transformed

    def __len__(self) -> int:
        return 46839
