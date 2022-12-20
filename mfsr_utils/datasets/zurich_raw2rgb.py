from dataclasses import dataclass
from pathlib import Path

import torchvision  # type: ignore[import]
from torch import Tensor
from torch.utils.data.dataset import Dataset
from typing_extensions import ClassVar

from mfsr_utils.datasets.protocols.downloadable import Downloadable


@dataclass
class ZurichRaw2Rgb(Dataset[Tensor], Downloadable):
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

    def __getitem__(self, index: int) -> Tensor:
        image_path = self.data_dir / self.dirname / "train" / "canon" / f"{index}.jpg"
        image_file: Tensor = torchvision.io.read_file(image_path.as_posix())
        image_jpg: Tensor = torchvision.io.decode_jpeg(image_file)
        return image_jpg

    def __len__(self) -> int:
        return 46839
