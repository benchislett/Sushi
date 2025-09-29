from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from sushi.utils import (
    check_color_rgb,
    check_image_rgb,
    check_image_shape,
    check_triangle_vertices,
)


def opencv_count_pixels_single(
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

    image_buffer = np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)
    cv2.fillPoly(image_buffer, [vertices.reshape((3, 1, 2))], 1.0)
    return int(np.sum(image_buffer > 0))


def opencv_draw_single_inplace(
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

    cv2.fillPoly(image, [vertices.reshape((3, 1, 2))], tuple(color.tolist()))


def opencv_draw_single(
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
    opencv_draw_single_inplace(image_copy, vertices, color)
    return image_copy


def opencv_drawloss_single(
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
    opencv_draw_single_inplace(modified_image, vertices, color)

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
