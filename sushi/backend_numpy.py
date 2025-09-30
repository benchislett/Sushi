from typing import ClassVar

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


def _pixel_array(image_shape: tuple[int, int]) -> NDArray[np.int32]:
    """Generate an array of pixel coordinates for an image of the given shape.

    Args:
        image_shape: The shape of the image, a tuple (H, W).

    Returns:
        An array of shape (H*W, 2) with dtype np.int32, where each row is the (x, y)
        coordinates of a pixel in the image.
    """
    H, W = image_shape
    y_coords, x_coords = np.meshgrid(
        np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing="ij"
    )
    pixel_coords = np.stack((x_coords.ravel(), y_coords.ravel()), axis=-1)
    return pixel_coords


def _points_in_triangle(
    points: NDArray[np.int32],
    vertices: NDArray[np.int32],
) -> NDArray[np.bool_]:
    """Check which points are inside a triangle defined by its vertices.

    Args:
        points: An array of shape (N, 2) with dtype np.int32 representing the (x, y)
            coordinates of the points to check.
        vertices: The vertices of the triangle, an array of shape (3, 2)
            with dtype np.int32 representing the (x, y) coordinates of the
            triangle's corners, with the origin at the top-left corner of the image.

    Returns:
        A boolean array of shape (N,) where each element is True if the corresponding
        point is inside the triangle, and False otherwise.
    """
    p0 = vertices[0]
    p1 = vertices[1]
    p2 = vertices[2]

    v0 = p2 - p0
    v1 = p1 - p0
    v2 = points - p0

    # Scalar dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot11 = np.dot(v1, v1)
    # Vector dot products
    dot20 = np.dot(v2, v0)
    dot21 = np.dot(v2, v1)

    # Calculate the denominator for the barycentric coordinates
    # This is related to the area of the triangle
    # if it's zero, the triangle is degenerate
    denominator = dot00 * dot11 - dot01 * dot01

    # A denominator of zero means the vertices are collinear, so no points are "inside"
    if denominator == 0:
        return np.zeros(len(points), dtype=bool)
    u_scaled = dot11 * dot20 - dot01 * dot21
    v_scaled = dot00 * dot21 - dot01 * dot20
    if denominator < 0:
        return (u_scaled <= 0) & (v_scaled <= 0) & (u_scaled + v_scaled >= denominator)
    else:
        return (u_scaled >= 0) & (v_scaled >= 0) & (u_scaled + v_scaled <= denominator)


def composit_over(
    foreground: NDArray[np.uint8],
    background: NDArray[np.uint8],
    alpha: float,
) -> NDArray[np.uint8]:
    """Composite a foreground color over a background color using the given alpha.

    Args:
        foreground: The foreground color, an array of shape (N,3) with dtype np.uint8
        background: The background color, an array of shape (N,3) with dtype np.uint8
        alpha: The alpha value for the foreground color, a float in [0.0, 1.0].
    Returns:
        The composited color, an array of shape (N,3) with dtype np.uint8.
    """
    return (foreground * alpha + background * (1 - alpha)).astype(np.uint8)


class NumpyRasterBackend(RasterBackend):
    name: ClassVar[str] = "numpy"

    @classmethod
    def triangle_count_pixels_single(
        cls: type["NumpyRasterBackend"],
        image_shape: tuple[int, int],
        vertices: NDArray[np.int32],
    ) -> int:
        check_image_shape(image_shape)
        check_triangle_vertices(vertices)
        pixel_coords = _pixel_array(image_shape)
        inside_mask = _points_in_triangle(pixel_coords, vertices)
        num_colored_pixels = int(np.sum(inside_mask))
        return num_colored_pixels

    @classmethod
    def triangle_draw_single_rgb_inplace(
        cls: type["NumpyRasterBackend"],
        image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        check_image_rgb(image)
        check_triangle_vertices(vertices)
        check_color_rgb(color)

        pixel_coords = _pixel_array(image.shape[:2])
        inside_mask = _points_in_triangle(pixel_coords, vertices)
        image.reshape(-1, 3)[inside_mask] = color

    @classmethod
    def triangle_draw_single_rgba_inplace(
        cls: type["NumpyRasterBackend"],
        image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        check_image_rgb(image)
        check_triangle_vertices(vertices)
        check_color_rgba(color)

        pixel_coords = _pixel_array(image.shape[:2])
        inside_mask = _points_in_triangle(pixel_coords, vertices)
        alpha = color[3] / 255.0
        image.reshape(-1, 3)[inside_mask] = composit_over(
            np.tile(color[:3], (np.sum(inside_mask), 1)),
            image.reshape(-1, 3)[inside_mask],
            alpha,
        )
