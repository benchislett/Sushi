from typing import Union

import numpy as np
from numpy.typing import NDArray


def check_image_shape(shape: Union[tuple[int, int], tuple[int, int, int]]) -> None:
    assert isinstance(shape, tuple)
    assert len(shape) in (2, 3)
    if len(shape) >= 2:
        assert shape[0] > 0
        assert shape[1] > 0
    if len(shape) == 3:
        assert shape[2] == 3  # 3-channel RGB


def check_image_rgb(image: NDArray[np.uint8]) -> None:
    assert image.ndim == 3
    assert image.shape[2] == 3  # 3-channel RGB
    assert image.dtype == np.uint8  # uint8 RGB values


def check_color_rgba(color: NDArray[np.uint8]) -> None:
    assert color.shape == (4,)
    assert color.dtype == np.uint8  # uint8 RGBA values


def check_triangle_vertices(vertices: NDArray[np.int32]) -> None:
    assert vertices.shape == (3, 2)
    assert vertices.dtype == np.int32  # int32 pixel coordinates


def np_image_mse(
    image1: NDArray[np.uint8],
    image2: NDArray[np.uint8],
) -> float:
    """Compute the mean squared error between two images.

    Args:
        image1: The first image, an array of shape (H, W, 3) with dtype np.uint8.
        image2: The second image, an array of shape (H, W, 3) with dtype np.uint8.

    Returns:
        The mean squared error between the two images.
    """
    check_image_rgb(image1)
    check_image_rgb(image2)
    assert image1.shape == image2.shape

    mse = np.mean((image1.astype(np.int64) - image2.astype(np.int64)) ** 2)
    return float(mse)
