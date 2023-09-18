from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, TypeVar

import torch
import torchvision  # type: ignore[import]
from torch import Tensor, nn
from torch.utils.data.dataset import Dataset

from mfsr_utils.datasets.protocols.downloadable import Downloadable

_T = TypeVar("_T")


# TODO: Do I need to normalize the images or convert them to floats?
# TODO: Document the type of the returned tensor.
@dataclass
class NTIRESyntheticBurstValidation2022(Dataset[_T], Downloadable):
    """Synthetic burst validation set introduced in [1]. The validation burst have been generated
    using a synthetic data generation pipeline. The dataset can be downloaded from
    https://data.vision.ee.ethz.ch/bhatg/SyntheticBurstVal.zip

    [1] Deep Burst Super-Resolution. Goutam Bhat, Martin Danelljan, Luc Van Gool, and Radu
    Timofte. CVPR 2021
    """

    url: ClassVar[str] = "https://data.vision.ee.ethz.ch/bhatg/SyntheticBurstVal.zip"
    filename: ClassVar[str] = "ntire_synburst_validation_2022.zip"
    dirname: ClassVar[str] = "ntire_synburst_validation_2022"
    mirrors: ClassVar[list[str]] = ["https://storage.googleapis.com/bsrt-supplemental/SyntheticBurstVal.zip"]

    data_dir: Path
    burst_size: int = 14
    transform: Callable[[Tensor, Tensor], _T] = nn.Identity()

    def __post_init__(self) -> None:
        assert (
            1 <= self.burst_size and self.burst_size <= 14
        ), "Only burst size in [1,14] are supported (there are 14 images in the burst)"

    def _read_burst(self, index: int) -> Tensor:
        """
        Args:
            index (int): Index of the burst to be returned. Must be in the range [0, 300).

        Returns:
            burst (Tensor): A tensor of shape [burst_size, 4, 48, 48].

                The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer
                mosaick.
        """
        image_pngs: list[Tensor] = []
        for i in range(self.burst_size):
            image_path = Path(self.data_dir) / self.dirname / "bursts" / f"{index:04}" / f"im_raw_{i:02}.png"
            image_file: Tensor = torchvision.io.read_file(image_path.as_posix())

            # TODO: Do we need to explicitly state the image read mode? These are bayered images
            # and we don't want them to be interpreted as RGB PNG files with an alpha layer.
            image_png: Tensor = torchvision.io.decode_png(image_file)
            image_pngs.append(image_png)

        stacked = torch.stack(image_pngs)
        # im_t = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1).float() / 2**14
        return stacked

    def _read_gt(self, index: int) -> Tensor:
        """
        Args:
            index (int): Index of the ground truth image to be returned. Must be in the range
                [0, 300).

        Returns:
            gt (Tensor): A tensor of shape [3, 384, 384].
        """
        image_path = Path(self.data_dir) / self.dirname / "gt" / f"{index:04}" / "im_rgb.png"
        image_file: Tensor = torchvision.io.read_file(image_path.as_posix())
        image_png: Tensor = torchvision.io.decode_png(image_file)
        # gt_t = torch.from_numpy(gt.astype(np.float32)).permute(2, 0, 1).float() / 2**14
        return image_png

    # TODO(@connorbaker): Update the docstring
    def __getitem__(self, index: int) -> _T:
        """
        Args:
            index (int): Index of the burst and ground truth image to be returned. Must be in the
                range [0, 300).

        Returns:
            A tensor of shape [burst_size, 4, 48, 48] and a tensor of shape [3, 384, 384]. The
            first tensor contains the burst and the second tensor contains the ground truth image.
            The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
            burst: LR RAW burst, a torch tensor of shape [burst_size, 4, 48, 48].

                The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer
                mosaick.
            gt : Ground truth linear image.
        """
        burst = self._read_burst(index)
        gt = self._read_gt(index)
        transformed: _T = self.transform(burst, gt)
        return transformed

    def __len__(self) -> int:
        return 300
