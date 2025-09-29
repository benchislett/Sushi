from typing import Optional

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from sushi.utils import (
    check_color_rgba,
    check_image_rgb,
    check_image_shape,
    check_triangle_vertices,
)


def _draw_triangle_over(
    image: Image.Image,
    vertices: NDArray[np.int32],
    color: NDArray[np.uint8],
) -> None:
    """Draw a triangle over the image.

    Args:
        image: The pillow image to draw on. This image is modified in place.
        vertices: The vertices of the triangle, an array of shape (3, 2)
            with dtype np.int32 representing the (x, y) coordinates of the
            triangle's corners, with the origin at the top-left corner of the image.
        color: The color of the triangle, as RGBA.
    """
    draw = ImageDraw.Draw(image, "RGBA")
    draw.polygon([tuple(v) for v in vertices], fill=tuple(color))


def pillow_count_pixels_single(
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

    # Use binary (1-bit) image mode to count the pixels.
    image_pil = Image.new("1", (image_shape[1], image_shape[0]), 0)
    draw = ImageDraw.Draw(image_pil, "1")
    draw.polygon([tuple(v) for v in vertices], fill=1)

    image_array = np.array(image_pil, dtype=np.bool_)
    num_colored_pixels = int(np.sum(image_array))
    return num_colored_pixels


def pillow_draw_single(
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
        color: The color of the triangle, an array of shape (4,) with dtype np.uint8,
            representing the RGBA color of the triangle.

    Returns:
        The modified image with the triangle drawn on it.
    """
    check_image_rgb(image)
    check_triangle_vertices(vertices)
    check_color_rgba(color)

    image_copy = image.copy()
    image_pil = Image.fromarray(image_copy)
    _draw_triangle_over(image_pil, vertices, color)
    return np.array(image_pil, dtype=np.uint8)


def pillow_drawloss_single(
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
        color: The color of the triangle, an array of shape (4,) with dtype np.uint8,
            representing the RGBA color of the triangle.
        base_loss: If provided, the MSE loss between the base and the target image.
            If not provided, it will be computed.

    Returns:
        The MSE loss delta `x` such that:
        `MSE(draw(triangle, image), target_image) == MSE(image, target_image) + x`.
    """
    check_image_rgb(image)
    check_image_rgb(target_image)
    check_triangle_vertices(vertices)
    check_color_rgba(color)

    image_copy_pil = Image.fromarray(image.copy())

    _draw_triangle_over(image_copy_pil, vertices, color)

    modified_image = np.array(image_copy_pil, dtype=np.uint8)

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
