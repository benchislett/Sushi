from typing import ClassVar, Optional

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from sushi.utils import (
    RasterBackend,
    check_color_rgb,
    check_color_rgba,
    check_image_rgb,
    check_image_shape,
    check_triangle_vertices,
)


class PillowRasterBackend(RasterBackend):
    name: ClassVar[str] = "pillow"

    @classmethod
    def triangle_count_pixels_single(
        cls: type["PillowRasterBackend"],
        image_shape: tuple[int, int],
        vertices: NDArray[np.int32],
    ) -> int:
        check_image_shape(image_shape)
        check_triangle_vertices(vertices)

        # Use binary (1-bit) image mode to count the pixels.
        image_pil = Image.new("1", (image_shape[1], image_shape[0]), 0)
        draw = ImageDraw.Draw(image_pil, "1")
        draw.polygon([tuple(v) for v in vertices], fill=1)

        image_array = np.array(image_pil, dtype=np.bool_)
        num_colored_pixels = int(np.sum(image_array))
        return num_colored_pixels

    @classmethod
    def triangle_draw_single_rgb_inplace(
        cls: type["PillowRasterBackend"],
        image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        check_image_rgb(image)
        check_triangle_vertices(vertices)
        check_color_rgb(color)

        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil, "RGB")
        draw.polygon([tuple(v) for v in vertices], fill=tuple(color))
        image[:, :, :] = np.array(image_pil, dtype=np.uint8)

    @classmethod
    def triangle_draw_single_rgba_inplace(
        cls: type["PillowRasterBackend"],
        image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        check_image_rgb(image)
        check_triangle_vertices(vertices)
        check_color_rgba(color)

        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil, "RGBA")
        draw.polygon([tuple(v) for v in vertices], fill=tuple(color))
        image[:, :, :] = np.array(image_pil, dtype=np.uint8)
