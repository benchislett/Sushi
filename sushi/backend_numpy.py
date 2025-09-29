from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from sushi.utils import (
    check_color_rgb,
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
    points_float = points.astype(np.float64)
    vertices_float = vertices.astype(np.float64)
    # This implementation uses the Barycentric coordinate system. A point P is inside
    # a triangle defined by vertices A, B, and C if it can be expressed as:
    # P = w*A + u*B + v*C, where w, u, v are non-negative and w + u + v = 1.
    #
    # We can rewrite this using two vectors originating from one vertex (A):
    # P - A = u*(B - A) + v*(C - A)
    #
    # This vector equation can be solved for u and v. The point is inside the
    # triangle if u >= 0, v >= 0, and u + v <= 1.

    p0 = vertices_float[0]
    p1 = vertices_float[1]
    p2 = vertices_float[2]

    # Define the triangle's edge vectors originating from p0
    v0 = p2 - p0
    v1 = p1 - p0
    # Define vectors from p0 to each point to be checked
    v2 = points_float - p0

    # Compute dot products to solve the system
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot11 = np.dot(v1, v1)
    # Vectorized dot product for all points
    dot20 = np.dot(v2, v0)
    dot21 = np.dot(v2, v1)

    # Calculate the denominator for the barycentric coordinates
    # This is related to the area of the triangle
    # if it's zero, the triangle is degenerate
    denominator = dot00 * dot11 - dot01 * dot01

    # A denominator of zero means the vertices are collinear, so no points are "inside"
    if denominator == 0:
        return np.zeros(len(points), dtype=bool)

    # Calculate barycentric coordinates u and v
    inv_denom = 1.0 / denominator
    u = (dot11 * dot20 - dot01 * dot21) * inv_denom
    v = (dot00 * dot21 - dot01 * dot20) * inv_denom

    # The point is inside if u, v, and w (where w = 1-u-v) are all >= 0.
    # This simplifies to checking if u >= 0, v >= 0, and u + v <= 1.
    return (u >= 0) & (v >= 0) & (u + v <= 1)


def numpy_count_pixels_single(
    image_shape: tuple[int, int],
    vertices: NDArray[np.int32],
) -> int:
    """Count the number of pixels that would be drawn over by a triangle on an image
    of a given shape.

    Args:
        image_shape: The shape of the image, a tuple (H, W).
        vertices: The vertices of the triangle, an array of shape (3, 2)
            with dtype np.int32 representing the (x, y) coordinates of the
            triangle's corners, with the origin at the top-left corner of the image.

    Returns:
        The number of pixels that would be colored when the triangle is drawn.
    """
    check_image_shape(image_shape)
    check_triangle_vertices(vertices)
    pixel_coords = _pixel_array(image_shape)
    inside_mask = _points_in_triangle(pixel_coords, vertices)
    num_colored_pixels = int(np.sum(inside_mask))
    return num_colored_pixels


def numpy_draw_single_inplace(
    image: NDArray[np.uint8],
    vertices: NDArray[np.int32],
    color: NDArray[np.uint8],
) -> None:
    """Draw a triangle over a given image. The input image is modified in place.

    Args:
        image: The base image, an array of shape (H, W, 3) with dtype np.uint8.
        vertices: The vertices of the triangle, an array of shape (3, 2)
            with dtype np.int32 representing the (x, y) coordinates of the
            triangle's corners, with the origin at the top-left corner of the image.
        color: The color of the triangle, an array of shape (3,) with dtype np.uint8,
            representing the RGB color of the triangle.
    """
    check_image_rgb(image)
    check_triangle_vertices(vertices)
    check_color_rgb(color)

    pixel_coords = _pixel_array(image.shape[:2])
    inside_mask = _points_in_triangle(pixel_coords, vertices)
    image.reshape(-1, 3)[inside_mask] = color


def numpy_draw_single(
    image: NDArray[np.uint8],
    vertices: NDArray[np.int32],
    color: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Draw a triangle over a given image. The input image is unmodified.

    Args:
        image: The base image, an array of shape (H, W, 3) with dtype np.uint8.
        vertices: The vertices of the triangle, an array of shape (3, 2)
            with dtype np.int32 representing the (x, y) coordinates of the
            triangle's corners, with the origin at the top-left corner of the image.
        color: The color of the triangle, an array of shape (3,) with dtype np.uint8,
            representing the RGB color of the triangle.

    Returns:
        The modified image with the triangle drawn on it.
    """
    check_image_rgb(image)
    check_triangle_vertices(vertices)
    check_color_rgb(color)

    image_copy = image.copy()
    numpy_draw_single_inplace(image_copy, vertices, color)
    return image_copy


def numpy_drawloss_single(
    image: NDArray[np.uint8],
    target_image: NDArray[np.uint8],
    vertices: NDArray[np.int32],
    color: NDArray[np.uint8],
    base_loss: Optional[float] = None,
) -> float:
    """Calculate the MSE loss delta that would be incurred by drawing a triangle
    over a given image, compared to a target image. The input image is unmodified.

    Args:
        image: The base image, an array of shape (H, W, 3) with dtype np.uint8.
        target_image: The target image, an array of shape (H, W, 3) with dtype np.uint8.
        vertices: The vertices of the triangle, an array of shape (3, 2)
            with dtype np.int32 representing the (x, y) coordinates of the
            triangle's corners, with the origin at the top-left corner of the image.
        color: The color of the triangle, an array of shape (3,) with dtype np.uint8,
            representing the RGB color of the triangle.
        base_loss: If provided, the MSE loss between the base and the target image.
            If not provided, it will be computed.

    Returns:
        The MSE loss delta `x` such that:
        `MSE(draw(triangle, image), target_image) == MSE(image, target_image) + x`.
    """
    check_image_rgb(image)
    check_image_rgb(target_image)
    check_triangle_vertices(vertices)
    check_color_rgb(color)

    modified_image = image.copy()
    numpy_draw_single_inplace(modified_image, vertices, color)

    # Compute the MSE loss between the modified image and the target image.
    if base_loss is None:
        base_loss = float(
            np.mean((image.astype(np.int64) - target_image.astype(np.int64)) ** 2)
        )

    assert base_loss is not None
    assert base_loss >= 0.0, "Base MSE should be non-negative."

    modified_loss = np.mean(
        (modified_image.astype(np.int64) - target_image.astype(np.int64)) ** 2
    )

    loss_delta = modified_loss - base_loss

    return float(loss_delta)
