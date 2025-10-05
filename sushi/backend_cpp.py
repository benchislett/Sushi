from typing import ClassVar, Optional

import numpy as np
from numpy.typing import NDArray

from sushi.utils import (
    RasterBackend,
    check_color_rgba,
    check_image_rgb,
    check_triangle_vertices,
)

try:
    from sushi_core import CPPRasterBackend as CoreBackend
except ImportError as e:
    raise ImportError(
        "C++ backend not available. Make sure to build the C++ extension first. "
        "Run 'pip install -e .' to build the extension."
    ) from e


class CPPRasterBackend(RasterBackend):
    """C++ implementation of triangle rasterization backend using nanobind."""

    name: ClassVar[str] = "cpp"

    @classmethod
    def triangle_draw_single_rgba_inplace(
        cls: type["CPPRasterBackend"],
        image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        """Draw a triangle with alpha blending over a given image.
        The input image is modified in place.

        Args:
            image: The base RGB image, an array of shape (H, W, 3) with dtype np.uint8.
            vertices: The vertices of the triangle, an array of shape (3, 2)
                with dtype np.int32 representing the (x, y) coordinates of the
                triangle's corners, with the origin at the top-left corner of the image.
            color: The color of the triangle, an array of shape (4,) with
                dtype np.uint8, representing the RGBA color of the triangle.
        """
        check_image_rgb(image)
        check_triangle_vertices(vertices)
        check_color_rgba(color)

        # Call the C++ implementation
        CoreBackend.triangle_draw_single_rgba_inplace(image, vertices, color)

    @classmethod
    def triangle_drawloss_batch_rgba(
        cls: type["RasterBackend"],
        image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        colors: NDArray[np.uint8],
        base_loss: Optional[float] = None,
    ) -> NDArray[np.float32]:
        check_image_rgb(image)
        check_image_rgb(target_image)
        check_triangle_vertices(vertices[0])
        check_color_rgba(colors[0])

        out = np.empty((vertices.shape[0],), dtype=np.float32)
        CoreBackend.triangle_drawloss_batch_rgba(
            image, target_image, vertices, colors, out
        )
        return out

    @classmethod
    def triangle_drawloss_single_rgba(
        cls: type["RasterBackend"],
        image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
        base_loss: Optional[float] = None,
    ) -> float:
        return cls.triangle_drawloss_batch_rgba(
            image, target_image, vertices[None, :, :], color[None, :]
        )[0].item()
