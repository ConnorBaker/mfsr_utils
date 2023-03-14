import math
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Type, TypedDict

import torch
from torch import Tensor, nn
from torchvision.transforms import (  # type: ignore[import]
    CenterCrop,
    InterpolationMode,
    RandomCrop,
)
from torchvision.transforms import functional as TF  # type: ignore[import]
from typing_extensions import Self

# TODO: Refactor to use the new mosaic implementation
from mfsr_utils.pipelines import camera
from mfsr_utils.pipelines.image_processing_params import ImageProcessingParams
from mfsr_utils.pipelines.image_transform_params import ImageTransformParams
from mfsr_utils.pipelines.meta_info import MetaInfo

# TODO: Find a clean way to know which device to allocate the tensors on.


def patches_from_image(image: Tensor, PH: int, PW: int) -> Tensor:
    """
    Takes an image and returns a tensor of patches.

    Args:
        image (Tensor): A tensor of at least three dimensions with the last two being the height
            and width of the image.
        PH (int): The height of the patches.
        PW (int): The width of the patches.

    Returns:
        A tensor of shape (*image.shape[:-2], NHP, NWP, PH, PW), where NHP is the number of
        patches in the height dimension, and NWP is the number of patches in the width dimension.
    """
    # Number of dimensions
    _ND: int = image.dim()
    assert _ND >= 3, (
        f"Image should have at least 3 dimensions, but has {_ND}; consider using torch.unsqueeze"
        " to add a batch dimension."
    )
    # Height
    _H: int = image.shape[-2]
    # Width
    _W: int = image.shape[-1]
    # Patch Height, Patch Width
    patches = (
        image
        # Unfolds along the height dimension, adding a dimension at the end
        .unfold(-2, PH, PH)
        # Because we have a new dimension at the end, -2 now refers to the width dimension
        .unfold(-2, PW, PW)
    )
    # Number of patches in the height dimension
    _NHP: int = patches.shape[-4]
    # Number of patches in the width dimension
    _NWP: int = patches.shape[-3]
    # Patch Height
    _PH: int = patches.shape[-2]
    # Patch Width
    _PW: int = patches.shape[-1]

    assert (
        _H // PH == _NHP
    ), f"Number of patches in the height dimension should be {_H // PH}, but is {_NHP}"
    assert (
        _W // PW == _NWP
    ), f"Number of patches in the width dimension should be {_W // PW}, but is {_NWP}"
    assert PH == _PH, f"Patch height should be {PH}, but is {_PH}"
    assert PW == _PW, f"Patch width should be {PW}, but is {_PW}"
    return patches


def image_from_patches(patches: Tensor) -> Tensor:
    """
    Takes a tensor of patches and returns an image.

    Args:
        patches (Tensor): A tensor of at least five dimensions with the last four being (NHP, NWP,
            PH, PW), where NHP is the number of patches in the height dimension, and NWP is the
            number of patches in the width dimension.

    Returns:
        A tensor of shape (*patches.shape[:-4], H, W), where H is the height of the image (given
        by NHP * PH), and W is the width of the image (given by NWP * PW).
    """
    # Number of dimensions
    _ND: int = patches.dim()
    assert _ND >= 5, (
        f"Image should have at least 5 dimensions, but has {_ND}; consider using torch.unsqueeze"
        " to add a batch dimension."
    )
    # Number of patches in the height dimension
    _NHP: int = patches.shape[-4]
    # Number of patches in the width dimension
    _NWP: int = patches.shape[-3]
    # Patch Height
    _PH: int = patches.shape[-2]
    # Patch Width
    _PW: int = patches.shape[-1]
    # Height
    _H: int = _NHP * _PH
    # Width
    _W: int = _NWP * _PW

    return (
        patches
        # We need to permute the dimensions so NHP is next to PH, and NWP is next to PW
        # (..., NHP, NWP, PH, PW) -> (..., NHP, PH, NWP, PW)
        .permute(*range(_ND - 4), _ND - 4, _ND - 2, _ND - 3, _ND - 1).reshape(
            *patches.shape[:-4], _H, _W
        )
    )


@dataclass
class PatchesInfo:
    """
    A class to hold the information about tensors of patches extracted from an image. Doesn't hold
    information about the channels or the patches themselves.

    Attributes:
        PH (int): The height of the patches.
        PW (int): The width of the patches.
        NHP (int): The number of patches in the height dimension.
        NWP (int): The number of patches in the width dimension.
        H (int): The height of the image (given by NHP * PH).
        W (int): The width of the image (given by NWP * PW).
    """

    PH: int
    PW: int
    NHP: int
    NWP: int
    H: int = field(init=False)
    W: int = field(init=False)

    def __post_init__(self: Self) -> None:
        self.H = self.NHP * self.PH  # type: ignore[assignment]
        self.W = self.NWP * self.PW  # type: ignore[assignment]

    @classmethod
    def of(cls: Type[Self], image: Tensor) -> Self:
        """
        Creates a PatchesInfo object from an image tensor.

        Args:
            image (Tensor): A tensor where the last four dimensions are (NHP, NWP, PH, PW), where
                NHP is the number of patches in the height dimension, NWP is the number of patches
                in the width dimension, PH is the patch height, and PW is the patch width.

        Returns:
            A PatchesInfo object.
        """
        NHP = image.shape[-4]
        NWP = image.shape[-3]
        PH = image.shape[-2]
        PW = image.shape[-1]

        return cls(PH, PW, NHP, NWP)


def rgb2rawburst(
    image: Tensor,
    burst_size: int,
    downsample_factor: float = 1,
    burst_transform_params: None | ImageTransformParams = None,
    image_processing_params: None | ImageProcessingParams = None,
) -> tuple[Tensor, Tensor, Tensor, MetaInfo]:
    """Generates a synthetic LR RAW burst from the input image. The input sRGB image is first
    converted to linear sensor space using an inverse camera pipeline. A LR burst is then
    generated by applying random transformations defined by burst_transform_params to the
    input image, and downsampling it by the downsample_factor. The generated burst is then
    mosaicekd and corrputed by random noise.
    """

    if image_processing_params is None:
        image_processing_params = ImageProcessingParams()

    image_processing_params.cam2rgb = image_processing_params.cam2rgb.to(
        dtype=image.dtype, device=image.device
    )

    image_processing_params.rgb2cam = image_processing_params.rgb2cam.to(
        dtype=image.dtype, device=image.device
    )

    # Approximately inverts global tone mapping.
    if image_processing_params.smoothstep:
        image = camera.invert_smoothstep(image)

    # Inverts gamma compression.
    if image_processing_params.compress_gamma:
        image = camera.gamma_expansion(image)

    # Inverts color correction.
    image = camera.apply_ccm(image, image_processing_params.rgb2cam)

    # Sample gains
    # FIXME: This just makes the image VERY green.
    image = image_processing_params.gain(image)

    # Clip saturated pixels.
    image = image.clamp(0.0, 1.0)

    # Generate LR burst
    image_burst_rgb = single2lrburst(
        image=image,
        burst_size=burst_size,
        downsample_factor=downsample_factor,
        transform_params=burst_transform_params,
    )

    # mosaic
    image_burst = camera.mosaic(image_burst_rgb)

    # Add noise
    # TODO: This is only acceptable if Noises(0, 0) is the identity under apply. If it isn't, we
    #       need to move the definition of noises inside an if statement to avoid applying it when
    #       add_noise is False.
    image_burst = image_processing_params.noise(image_burst)

    # Clip saturated pixels.
    image_burst = image_burst.clamp(0.0, 1.0)

    meta_info = MetaInfo(
        rgb2cam=image_processing_params.rgb2cam,
        cam2rgb=image_processing_params.cam2rgb,
        gain=image_processing_params.gain,
        smoothstep=image_processing_params.smoothstep,
        compress_gamma=image_processing_params.compress_gamma,
        noise=image_processing_params.noise,
    )

    return image_burst, image, image_burst_rgb, meta_info


def single2lrburst(
    image: Tensor,
    burst_size: int,
    downsample_factor: float = 1.0,
    transform_params: None | ImageTransformParams = None,
) -> Tensor:
    """Generates a burst of size burst_size from the input image by applying random
    transformations defined by transform_params, and downsampling the resulting burst by
    downsample_factor.

    Args:
        image (Tensor): Input image. Must be of shape (C, H, W) and floating dtype.
        burst_size (int): Size of the burst to generate.
        downsample_factor (float, optional): Downsampling factor. Defaults to 1.0.
        transform_params (ImageTransformParams, optional): Parameters for the random
            transformations. Defaults to None.

    Returns:
        Tensor: The generated burst
    """
    if transform_params is None:
        transform_params = ImageTransformParams()

    # TODO: Some operations aren't supported with BFLOAT16, so we need to convert to FP32.
    original_dtype = image.dtype

    burst: list[Tensor] = []
    sample_pos_inv_all: list[Tensor] = []

    for i in range(burst_size):
        if i == 0:
            translate: list[int] = [0, 0]
            theta: float = 0.0
            shear: list[float] = [0.0, 0.0]
            scale: float = 1.0
        else:
            # Sample random image transformation parameters
            max_translation = transform_params.max_translation
            translate = [
                random.randint(-max_translation, max_translation),
                random.randint(-max_translation, max_translation),
            ]

            max_rotation = transform_params.max_rotation
            theta = random.uniform(-max_rotation, max_rotation)

            max_shear = transform_params.max_shear
            shear = [random.uniform(-max_shear, max_shear), random.uniform(-max_shear, max_shear)]

            max_ar = transform_params.max_ar
            ar_factor: float = math.exp(random.uniform(-max_ar, max_ar))

            max_scale = transform_params.max_scale
            scale = math.exp(random.uniform(-max_scale, max_scale)) * ar_factor

        if original_dtype == torch.bfloat16:
            image = image.to(torch.float32)

        image_transformed = TF.affine(
            image,
            angle=theta,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
        )

        if original_dtype == torch.bfloat16:
            image = image.to(torch.bfloat16)
            image_transformed = image_transformed.to(torch.bfloat16)

        if transform_params.border_crop > 0:
            border_crop = transform_params.border_crop
            image_transformed = TF.center_crop(
                image_transformed,
                [
                    image_transformed.shape[-2] - 2 * border_crop,
                    image_transformed.shape[-1] - 2 * border_crop,
                ],
            )

        # Downsample the image
        image_transformed = TF.resize(
            image_transformed,
            [
                int(image_transformed.shape[-2] / downsample_factor),
                int(image_transformed.shape[-1] / downsample_factor),
            ],
            interpolation=InterpolationMode.BILINEAR,
        )

        burst.append(image_transformed)
        sample_pos_inv_all.append(torch.zeros_like(image_transformed))

    burst_images = torch.stack(burst)

    return burst_images


class SyntheticBurstGeneratorData(TypedDict):
    """
    burst: Generated LR RAW burst, a torch tensor of shape

        [
            burst_size,
            4,
            self.crop_sz / (2*self.downsample_factor),
            self.crop_sz / (2*self.downsample_factor)
        ]

        The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
        The extra factor 2 in the denominator (2*self.downsample_factor) corresponds to the
        mosaicking operation.

    gt: The HR RGB ground truth in the linear sensor space, a torch tensor of shape
        [3, self.crop_sz, self.crop_sz]
    """

    burst: Tensor
    gt: Tensor


@dataclass(eq=False)
class SyntheticBurstGeneratorTransform(torch.nn.Module):
    """Synthetic burst dataset for joint denoising, demosaicking, and super-resolution. RAW Burst
    sequences are synthetically generated on the fly as follows. First, a single image is loaded
    from the base_dataset. The sampled image is converted to linear sensor space using the inverse
    camera pipeline employed in [1]. A burst sequence is then generated by adding random
    translations and rotations to the converted image. The generated burst is then converted is
    then mosaicked, and corrupted by random noise to obtain the RAW burst.

    [1] Unprocessing Images for Learned Raw Denoising, Brooks, Tim and Mildenhall, Ben and Xue,
    Tianfan and Chen, Jiawen and Sharlet, Dillon and Barron, Jonathan T, CVPR 2019
    """

    burst_size: int
    crop_sz: int
    dtype: None | torch.dtype = None
    device: None | torch.device = None
    final_crop_sz: int = field(init=False)
    downsample_factor: int = 4
    burst_transform_params: ImageTransformParams = field(
        default_factory=partial(
            ImageTransformParams,
            max_translation=24.0,
            max_rotation=1.0,
            max_shear=0.0,
            max_scale=0.0,
            border_crop=24,
        )
    )
    image_processing_params: ImageProcessingParams = field(
        default_factory=partial(
            ImageProcessingParams,
            compress_gamma=False,
            random_ccm=False,
            random_gain=False,
            random_noise=False,
            smoothstep=False,
        )
    )

    def __post_init__(self) -> None:
        super().__init__()
        self.final_crop_sz = self.crop_sz + 2 * self.burst_transform_params.border_crop
        self.random_crop: nn.Module = RandomCrop(self.final_crop_sz)
        self.post_crop: nn.Module = CenterCrop(self.crop_sz)

    def __call__(self, frame: Tensor) -> SyntheticBurstGeneratorData:  # type: ignore
        # Extract a random crop from the image
        frame = frame.to(dtype=self.dtype, device=self.device)
        prepared_frame: Tensor = self.random_crop(frame)

        burst, gt, _burst_rgb, _meta_info = rgb2rawburst(
            prepared_frame,
            burst_size=self.burst_size,
            downsample_factor=self.downsample_factor,
            burst_transform_params=self.burst_transform_params,
            image_processing_params=self.image_processing_params,
        )
        gt = self.post_crop(gt)

        return SyntheticBurstGeneratorData(
            burst=burst,
            gt=gt,
        )
