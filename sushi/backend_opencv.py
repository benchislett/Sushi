from typing import ClassVar

import cv2
import numpy as np
from numpy.typing import NDArray

from sushi.utils import (
    RasterBackend,
    check_color_rgb,
    check_color_rgba,
    check_image_rgb,
    check_image_shape,
    check_triangle_vertices,
)


class OpenCVRasterBackend(RasterBackend):
    name: ClassVar[str] = "opencv"

    @classmethod
    def triangle_count_pixels_single(
        cls: type["OpenCVRasterBackend"],
        image_shape: tuple[int, int],
        vertices: NDArray[np.int32],
    ) -> int:
        check_image_shape(image_shape)
        check_triangle_vertices(vertices)

        image_buffer = np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)
        cv2.fillPoly(image_buffer, [vertices.reshape((3, 1, 2))], 1.0)
        return int(np.sum(image_buffer > 0))

    @classmethod
    def triangle_draw_single_rgb_inplace(
        cls: type["OpenCVRasterBackend"],
        image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        check_image_rgb(image)
        check_triangle_vertices(vertices)
        check_color_rgb(color)

        cv2.fillPoly(image, [vertices.reshape((3, 1, 2))], tuple(color.tolist()))

    @classmethod
    def triangle_draw_single_rgba_inplace(
        cls: type["OpenCVRasterBackend"],
        image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        check_image_rgb(image)
        check_triangle_vertices(vertices)
        check_color_rgba(color)

        overlay = image.copy()
        cv2.fillPoly(overlay, [vertices.reshape((3, 1, 2))], tuple(color.tolist()))
        alpha = color[3] / 255.0
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
