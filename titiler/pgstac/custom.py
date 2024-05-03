"""LiDAR Change Detection Visualization"""
from typing import Sequence

import numpy as np
from pydantic import Field

from titiler.core.algorithm import BaseAlgorithm
from rio_tiler.models import ImageData
from rio_tiler.colormap import apply_cmap, cmap
from rio_tiler.utils import linear_rescale




# derived from .lyrx file
CHANGE_DETECTION_CMAP = [
    [[-12, -2], [164, 0, 0, 100]],
    [[-2, -1.65], [179, 43, 35, 100]],
    [[-1.65, -1.3], [194, 85, 69, 100]],
    [[-1.3, -0.95], [210, 128, 104, 100]],
    [[-0.95, -0.6], [225, 170, 139, 100]],
    [[-0.6, -0.25], [240, 213, 173, 100]],
    [[-0.25, 0.0], [255, 255, 255, 0]],
    [[0.0, 0.25], [255, 255, 255, 0]],
    [[0.25, 0.6], [204, 255, 255, 100]],
    [[0.6, 0.95], [170, 212, 238, 100]],
    [[0.95, 1.3], [136, 170, 220, 100]],
    [[1.3, 1.65], [102, 128, 203, 100]],
    [[1.65, 2], [68, 85, 186, 100]],
    [[2, 18.804758071899], [34, 43, 168, 100]],
]


class ChangeDetectionVisualize(BaseAlgorithm):
    """Visualize Enbridge LiDAR Change Detection"""

    title: str = "Change Detection Visualize"
    description: str = (
        "Render LiDAR Change Detection products as greyscale, with change values above or below treshold coloured."
    )

    # parameters
    rescale: list[float] = Field([-12, 18.804758071899], min_length=2, max_length=2)

    # metadata
    input_nbands: int = 1
    output_nbands: int = 1
    output_dtype: str = "uint8"
    output_min: Sequence[int] = 0
    output_max: Sequence[int] = 255

    def __call__(self, img: ImageData) -> ImageData:

        data = linear_rescale(
            img.data,
            in_range=(self.rescale[0], self.rescale[1]),
        ).astype("uint8")

        class_data, class_mask = apply_cmap(img.data, CHANGE_DETECTION_CMAP)
        class_mask = class_mask.astype("bool")
        data, _ = apply_cmap(data, cmap.get("greys"))
        data[:, class_mask] = class_data[:, class_mask]

        data = np.ma.MaskedArray(data)
        data.mask = ~img.mask
        return ImageData(data, crs=img.crs, bounds=img.bounds)
