import random
from dataclasses import dataclass, field
from typing import TypedDict

import cv2  # type: ignore[import]
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torchvision.transforms import (  # type: ignore[import]
    CenterCrop,
    ConvertImageDtype,
    RandomCrop,
)

# TODO: Refactor to use the new mosaic implementation
from mfsr_utils.pipelines.camera import (
    apply_ccm,
    gamma_expansion,
    invert_smoothstep,
    mosaic,
    random_ccm,
)
from mfsr_utils.pipelines.image_processing_params import ImageProcessingParams
from mfsr_utils.pipelines.image_transform_mats import get_transform_mat
from mfsr_utils.pipelines.image_transform_params import ImageTransformParams
from mfsr_utils.pipelines.meta_info import MetaInfo
from mfsr_utils.pipelines.noises import Noises
from mfsr_utils.pipelines.rgb_gains import RgbGains
from mfsr_utils.pipelines.types import InterpolationType

# TODO: Find a clean way to know which device to allocate the tensors on.


def rgb2rawburst(
    image: Tensor,
    burst_size: int,
    downsample_factor: float = 1,
    burst_transform_params: None | ImageTransformParams = None,
    image_processing_params: None | ImageProcessingParams = None,
    interpolation_type: InterpolationType = "bilinear",
) -> tuple[Tensor, Tensor, Tensor, Tensor, MetaInfo]:
    """Generates a synthetic LR RAW burst from the input image. The input sRGB image is first
    converted to linear sensor space using an inverse camera pipeline. A LR burst is then
    generated by applying random transformations defined by burst_transform_params to the
    input image, and downsampling it by the downsample_factor. The generated burst is then
    mosaicekd and corrputed by random noise.
    """

    if image_processing_params is None:
        image_processing_params = ImageProcessingParams()

    # Sample camera pipeline params
    if image_processing_params.random_ccm:
        rgb2cam = random_ccm(dtype=image.dtype, device=image.device)
    else:
        rgb2cam = torch.eye(3, dtype=image.dtype, device=image.device)
    cam2rgb = rgb2cam.inverse()

    # Approximately inverts global tone mapping.
    if image_processing_params.smoothstep:
        image = invert_smoothstep(image)

    # Inverts gamma compression.
    if image_processing_params.compress_gamma:
        image = gamma_expansion(image)

    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)

    # Sample gains
    # FIXME: This just makes the image VERY green.
    if image_processing_params.random_gains:
        # Approximately inverts white balance and brightening.
        gains = RgbGains.random_gains()
        image = gains.safe_invert_gains(image)
    else:
        gains = RgbGains(1.0, 1.0, 1.0)

    # Clip saturated pixels.
    image = image.clamp(0.0, 1.0)

    # Generate LR burst
    image_burst_rgb, flow_vectors = single2lrburst(
        image=image,
        burst_size=burst_size,
        downsample_factor=downsample_factor,
        transform_params=burst_transform_params,
        interpolation_type=interpolation_type,
    )

    # mosaic
    image_burst = mosaic(image_burst_rgb.clone())

    # Add noise
    if image_processing_params.add_noise:
        noises = Noises.random_noise_levels()
        image_burst = noises.apply(image_burst)
    else:
        noises = Noises(0.0, 0.0)

    # Clip saturated pixels.
    image_burst = image_burst.clamp(0.0, 1.0)

    meta_info = MetaInfo(
        rgb2cam=rgb2cam,
        cam2rgb=cam2rgb,
        gains=gains,
        smoothstep=image_processing_params.smoothstep,
        compress_gamma=image_processing_params.compress_gamma,
        noises=noises,
    )

    return image_burst, image, image_burst_rgb, flow_vectors, meta_info


def single2lrburst(
    image: Tensor,
    burst_size: int,
    downsample_factor: float = 1.0,
    transform_params: None | ImageTransformParams = None,
    interpolation_type: InterpolationType = "bilinear",
) -> tuple[Tensor, Tensor]:
    """Generates a burst of size burst_size from the input image by applying random
    transformations defined by transform_params, and downsampling the resulting burst by
    downsample_factor.

    Note: All transformations occur on the CPU (thanks OpenCV) with float64.

    Args:
        image (Tensor): Input image. Must be of shape (C, H, W) and floating dtype.
        burst_size (int): Size of the burst to generate.
        downsample_factor (float, optional): Downsampling factor. Defaults to 1.0.
        transform_params (ImageTransformParams, optional): Parameters for the random
            transformations. Defaults to None.
        interpolation_type (InterpolationType, optional): Interpolation type. Defaults to
            "bilinear".
        dtype (torch.dtype, optional): Data type of the output. Defaults to torch.float64.
        device (torch.device, optional): Device to allocate the output on. Defaults to
            torch.device("cpu").

    Returns:
        tuple[Tensor, Tensor]: The generated burst and the flow vectors.
    """
    if transform_params is None:
        transform_params = ImageTransformParams()

    interpolation: int
    match interpolation_type:
        case "nearest":
            interpolation = cv2.INTER_NEAREST  # type: ignore[attr-defined]
        case "bilinear":
            interpolation = cv2.INTER_LINEAR  # type: ignore[attr-defined]
        case "bicubic":
            interpolation = cv2.INTER_CUBIC  # type: ignore[attr-defined]
        case "lanczos":
            interpolation = cv2.INTER_LANCZOS4  # type: ignore[attr-defined]

    image_np: npt.NDArray[np.uint8] = (
        (image * 255.0).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
    )

    burst: list[Tensor] = []
    sample_pos_inv_all: list[Tensor] = []

    rvs, cvs = torch.meshgrid(
        [
            torch.arange(0, image_np.shape[0], dtype=torch.uint8, device="cpu"),
            torch.arange(0, image_np.shape[1], dtype=torch.uint8, device="cpu"),
        ],
        indexing="ij",
    )

    sample_grid = torch.stack((cvs, rvs, torch.ones_like(cvs)), dim=-1).to(
        dtype=torch.float64, device="cpu"
    )

    for i in range(burst_size):
        if i == 0:
            # For base image, do not apply any random transformations. We only translate the image
            # to center the sampling grid
            shift: float = (downsample_factor / 2.0) - 0.5
            translation: tuple[float, float] = (shift, shift)
            theta: float = 0.0
            shear_factor: tuple[float, float] = (0.0, 0.0)
            scale_factor_tuple: tuple[float, float] = (1.0, 1.0)
        else:
            # Sample random image transformation parameters
            max_translation = transform_params.max_translation

            if max_translation <= 0.01:
                shift = (downsample_factor / 2.0) - 0.5
                translation = (shift, shift)
            else:
                translation = (
                    random.uniform(-max_translation, max_translation),
                    random.uniform(-max_translation, max_translation),
                )

            max_rotation = transform_params.max_rotation
            theta = random.uniform(-max_rotation, max_rotation)

            max_shear = transform_params.max_shear
            shear_x = random.uniform(-max_shear, max_shear)
            shear_y = random.uniform(-max_shear, max_shear)
            shear_factor = (shear_x, shear_y)

            max_ar_factor = transform_params.max_ar_factor
            ar_factor: float = np.exp(random.uniform(-max_ar_factor, max_ar_factor))

            max_scale = transform_params.max_scale
            scale_factor: float = np.exp(random.uniform(-max_scale, max_scale))

            scale_factor_tuple = (
                scale_factor,
                scale_factor * ar_factor,
            )

        output_size: tuple[int, int] = (image_np.shape[1], image_np.shape[0])

        # Generate a affine transformation matrix corresponding to the sampled parameters
        transform_mat_tensor: Tensor = get_transform_mat(
            (image_np.shape[0], image_np.shape[1]),
            translation,
            theta,
            shear_factor,
            scale_factor_tuple,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        transform_mat_np: npt.NDArray[np.float64] = transform_mat_tensor.numpy()

        # Apply the sampled affine transformation
        image_np_t: npt.NDArray[np.uint8] = cv2.warpAffine(  # type: ignore
            image_np,
            transform_mat_np,
            output_size,
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,  # type: ignore
        )

        transform_mat_tensor_3x3: Tensor = torch.cat(
            (
                transform_mat_tensor,
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device="cpu").view(1, 3),
            ),
            dim=0,
        )
        transform_mat_tensor_inverse: Tensor = transform_mat_tensor_3x3.inverse()[
            :2, :
        ].contiguous()

        sample_pos_inv: Tensor = torch.mm(
            sample_grid.view(-1, 3), transform_mat_tensor_inverse.t()
        ).view(*sample_grid.shape[:2], -1)

        if transform_params.border_crop > 0:
            border_crop = transform_params.border_crop

            image_np_t = image_np_t[border_crop:-border_crop, border_crop:-border_crop, :]
            sample_pos_inv = sample_pos_inv[border_crop:-border_crop, border_crop:-border_crop, :]

        # Downsample the image
        image_np_t = cv2.resize(  # type: ignore
            image_np_t,
            None,  # type: ignore
            fx=1.0 / downsample_factor,  # type: ignore
            fy=1.0 / downsample_factor,  # type: ignore
            interpolation=interpolation,
        )
        sample_pos_inv = cv2.resize(  # type: ignore
            sample_pos_inv.numpy(),
            None,  # type: ignore
            fx=1.0 / downsample_factor,  # type: ignore
            fy=1.0 / downsample_factor,  # type: ignore
            interpolation=interpolation,
        )

        sample_pos_inv = (
            torch.from_numpy(sample_pos_inv).permute(2, 0, 1).contiguous()  # type: ignore
        )
        image_t_tensor = (
            torch.from_numpy(image_np_t)  # type: ignore[assignment]
            .permute(2, 0, 1)
            .to(torch.float64)
            / 255.0
        )

        burst.append(image_t_tensor)
        sample_pos_inv_all.append(sample_pos_inv / downsample_factor)

    burst_images = torch.stack(burst)
    sample_pos_inv_all_tensor = torch.stack(sample_pos_inv_all)

    # Compute the flow vectors to go from the i'th burst image to the base image
    flow_vectors = sample_pos_inv_all_tensor - sample_pos_inv_all_tensor[:, :1, ...]

    # Move everything to the appropriate device and dtype
    return burst_images, flow_vectors


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

    frame_gt: The HR RGB ground truth in the linear sensor space, a torch tensor of shape
        [3, self.crop_sz, self.crop_sz]

    flow_vectors: The ground truth flow vectors between a burst image and the base image (i.e. the
        first image in the burst).
        The flow_vectors can be used to warp the burst images to the base frame, using the 'warp'
        function in utils.warp package.
        flow_vectors is torch tensor of shape

        [
            burst_size,
            2,
            self.crop_sz / self.downsample_factor,
            self.crop_sz / self.downsample_factor
        ]

        Note that the flow_vectors are in the LR RGB space, before mosaicking. Hence it has twice
        the number of rows and columns, compared to the output burst.

        NOTE: The flow_vectors are only available during training for the purpose of using any
        auxiliary losses if needed. The flow_vectors will NOT be provided for the bursts in the
        test set

    meta_info: A dictionary containing the parameters used to generate the synthetic burst.
    """

    burst: Tensor
    gt: Tensor
    flow_vectors: Tensor
    # meta_info: MetaInfo


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
    dtype: torch.dtype
    final_crop_sz: int = field(init=False)
    downsample_factor: int = 4
    burst_transform_params: ImageTransformParams = ImageTransformParams(
        max_translation=24.0,
        max_rotation=1.0,
        max_shear=0.0,
        max_scale=0.0,
        border_crop=24,
    )
    image_processing_params: ImageProcessingParams = ImageProcessingParams(
        random_ccm=False,
        random_gains=False,
        smoothstep=False,
        compress_gamma=False,
        add_noise=False,
    )
    interpolation_type: InterpolationType = "bilinear"

    def __post_init__(self) -> None:
        super().__init__()
        self.final_crop_sz = self.crop_sz + 2 * self.burst_transform_params.border_crop
        self.prepare: nn.Module = nn.Sequential(
            ConvertImageDtype(torch.float32), RandomCrop(self.final_crop_sz)
        )
        self.post_crop: nn.Module = CenterCrop(self.crop_sz)
        self.post_dtype_converter: nn.Module = ConvertImageDtype(self.dtype)

    def __call__(self, frame: Tensor) -> SyntheticBurstGeneratorData:  # type: ignore
        # Extract a random crop from the image
        prepared_frame: Tensor = self.prepare(frame)

        burst, gt, _burst_rgb, flow_vectors, meta_info = rgb2rawburst(  # type: ignore
            prepared_frame,
            self.burst_size,
            self.downsample_factor,
            burst_transform_params=self.burst_transform_params,
            image_processing_params=self.image_processing_params,
            interpolation_type=self.interpolation_type,
        )
        burst = self.post_dtype_converter(burst)
        gt = self.post_dtype_converter(self.post_crop(gt))
        flow_vectors = self.post_dtype_converter(flow_vectors)

        return SyntheticBurstGeneratorData(
            burst=burst,
            gt=gt,
            flow_vectors=flow_vectors,
        )
