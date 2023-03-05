from dataclasses import dataclass


@dataclass
class ImageTransformParams:
    """Dataclass for storing transformation parameters"""

    border_crop: int = 0
    max_rotation: float = 0.0
    max_shear: float = 0.0
    max_ar: float = 0.0
    max_scale: float = 0.0
    max_translation: int = 0
