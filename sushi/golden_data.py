"""
This module generates sets of 2D triangles with various shapes, sizes,
and distributions on a canvas. It provides a flexible interface to create
sets of triangles for testing, visualization, and benchmarking.

This file can be invoked directly as a script to generate an image showing
a few example triangle sets, using Matplotlib for visualization. When running
as a script, it saves the output to 'triangle_examples.png'.

The main function is `generate_triangles()`, which accepts parameters to
control the number, shape, size, and distribution of triangles.
The generated triangles are returned as a NumPy array of shape (N, 3, 2),
where N is the number of triangles, and each triangle has three 2D vertices.
"""

import math
from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

# --- Constants and Mappings ---

# Defines the side length range in pixels for each size category.
SIZE_RANGES = {
    "micro": (4, 8),
    "tiny": (8, 16),
    "small": (16, 32),
    "medium": (32, 64),
    "large": (64, 128),
    "very_large": (128, 256),
    "huge": (256, 512),
    "gigantic": (512, 1024),
    "massive": (1024, 2048),
}

ShapeType = Literal[
    "equilateral", "right", "random", "wide", "tall", "slanted-up", "slanted-down"
]
SizeType = Literal[
    "micro",
    "tiny",
    "small",
    "medium",
    "large",
    "very_large",
    "huge",
    "gigantic",
    "massive",
]
DistributionType = Literal["uniform", "normal", "spaced", "center"]

# --- Core Generation Function ---


def generate_triangles(
    count: int,
    screen_width: int,
    screen_height: int,
    shape_type: ShapeType = "random",
    size: SizeType = "medium",
    distribution: DistributionType = "uniform",
    distribution_mean: Optional[tuple[float, float]] = None,
    distribution_std_dev: Optional[tuple[float, float]] = None,
    random_rotation: bool = True,
    random_seed: Optional[int] = 42,
    **kwargs: Any,
) -> NDArray[np.int32]:
    """
    Generates a collection of 2D triangles with integer coordinates.

    Args:
        count (int): The number of triangles to generate.
        screen_width (int): The width of the target screen/viewport.
        screen_height (int): The height of the target screen/viewport.
        shape_type (ShapeType): The type of triangle shape.
        size (SizeType): The approximate size of the triangles.
        distribution (DistributionType): The distribution of triangle centerpoints.
        distribution_mean (Optional[tuple[float, float]]): Mean (x, y) for 'normal'
            distribution. Defaults to screen center.
        distribution_std_dev (Optional[tuple[float, float]]): Std dev for 'normal'
            distribution. Defaults to 1/6th of screen dimensions.
        random_rotation (bool): If True, applies random rotation, jitter, and flipping
            to triangles.
        random_seed (Optional[int]): Seed for the random number generator for
            reproducible results. Defaults to 42, can be set to None for variability.

    Returns:
        np.ndarray: A NumPy array of shape (count, 3, 2) and dtype np.int32,
                    representing N triangles with three 2D vertices each.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if size not in SIZE_RANGES:
        raise ValueError(
            f"Invalid size '{size}'. Available sizes: {list(SIZE_RANGES.keys())}"
        )

    # 1. Generate center points to position each triangle
    if distribution == "uniform":
        centers = _generate_uniform_centers(count, screen_width, screen_height)
    elif distribution == "normal":
        centers = _generate_normal_centers(
            count,
            screen_width,
            screen_height,
            mean=distribution_mean,
            std_dev=distribution_std_dev,
        )
    elif distribution == "center":
        centers = np.array([[screen_width / 2, screen_height / 2]] * count)
    elif distribution == "spaced":
        centers = _generate_spaced_centers(count, screen_width, screen_height)
    else:
        raise ValueError(
            f"Invalid distribution '{distribution}'. "
            "Choose 'uniform', 'normal', or 'spaced'."
        )

    # 2. Generate rotation angles and sizes for each triangle
    angles: NDArray[np.float64] = np.zeros(count)
    full_rotate_shape_types = ["equilateral", "right", "random"]
    jitter_shape_types = ["wide", "tall", "slanted-up", "slanted-down"]
    if shape_type in full_rotate_shape_types and random_rotation:
        angles = np.random.uniform(0, 2 * np.pi, size=count)
    elif shape_type in jitter_shape_types:
        if random_rotation:
            jitter_amount = 0.1 * np.pi
            angles = np.random.uniform(-jitter_amount, jitter_amount, size=count)
        if shape_type == "tall":
            angles += np.pi / 2
        elif shape_type == "slanted-up":
            angles += -np.pi / 4  # y = -x diagonal
        elif shape_type == "slanted-down":
            angles += np.pi / 4  # y = x diagonal

    # 3. Generate sizes for each triangle
    min_len, max_len = SIZE_RANGES[size]
    lengths = np.random.uniform(min_len, max_len, size=count)

    # 4. Generate each triangle based on shape type and size
    base_tris: NDArray[np.float64]
    if shape_type == "equilateral":
        base_tris = _generate_equilateral_vectorized(lengths)
    elif shape_type == "right":
        base_tris = _generate_right_vectorized(lengths, lengths)
    elif shape_type == "random":
        base_tris = _generate_random_vectorized(lengths / 2)  # Use length as radius
    elif shape_type in ["wide", "tall", "slanted-up", "slanted-down"]:
        base_tris = _generate_long_triangle_vectorized(lengths)
        # Apply flipping for wide/tall/slanted shapes
        if random_rotation:
            flip_mask = np.random.rand(count) < 0.5
            base_tris[flip_mask, :, 0] *= -1

    # 5. Apply rotation and translation to each triangle
    cosines = np.cos(angles)
    sines = np.sin(angles)
    rot_matrices_T = np.array([[cosines, sines], [-sines, cosines]]).transpose(2, 0, 1)
    rotated_tris = base_tris @ rot_matrices_T  # Shape: (count, 3, 2)
    translated_tris = rotated_tris + centers[:, np.newaxis, :]

    return np.array(translated_tris, dtype=np.int32)


# --- Helper Functions for Distributions ---


def _generate_uniform_centers(
    count: int, width: int, height: int
) -> NDArray[np.float64]:
    """Generates center points uniformly across the screen."""
    centers_x = np.random.uniform(0, width, size=(count, 1))
    centers_y = np.random.uniform(0, height, size=(count, 1))
    return np.hstack([centers_x, centers_y])


def _generate_normal_centers(
    count: int,
    width: int,
    height: int,
    mean: Optional[tuple[float, float]] = None,
    std_dev: Optional[tuple[float, float]] = None,
) -> NDArray[np.float64]:
    """Generates center points normally distributed around a mean."""
    if mean is None:
        mean = (width / 2, height / 2)
    if std_dev is None:
        std_dev = (width / 6, height / 6)

    centers = np.random.normal(loc=mean, scale=std_dev, size=(count, 2))
    # Clip values to ensure they stay within the screen bounds
    centers[:, 0] = np.clip(centers[:, 0], 0, width)
    centers[:, 1] = np.clip(centers[:, 1], 0, height)
    return centers


def _generate_spaced_centers(
    count: int, width: int, height: int
) -> NDArray[np.float64]:
    """Generates center points that are roughly evenly spaced in a grid."""
    if count == 0:
        return np.array([])

    # Determine grid dimensions to be as square as possible
    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / cols))

    # Generate grid points
    x_points = np.linspace(0.5 / cols, 1 - 0.5 / cols, cols) * width
    y_points = np.linspace(0.5 / rows, 1 - 0.5 / rows, rows) * height

    xx, yy = np.meshgrid(x_points, y_points)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    # If the grid is larger than count, shuffle and pick `count` points
    np.random.shuffle(grid_points)
    return grid_points[:count]


# --- Helper Functions for Triangle Shapes (centered at origin) ---


def _generate_equilateral_vectorized(side: NDArray[np.float64]) -> NDArray[np.float64]:
    """Generates equilateral triangles for an array of side lengths."""
    height = side * np.sqrt(3) / 2
    v1 = np.stack([np.zeros_like(side), 2 * height / 3], axis=-1)
    v2 = np.stack([-side / 2, -height / 3], axis=-1)
    v3 = np.stack([side / 2, -height / 3], axis=-1)
    return np.stack([v1, v2, v3], axis=1)


def _generate_right_vectorized(
    width: NDArray[np.float64], height: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Generates right triangles for arrays of widths and heights."""
    v1 = np.stack([-width / 3, -height / 3], axis=-1)
    v2 = np.stack([2 * width / 3, -height / 3], axis=-1)
    v3 = np.stack([-width / 3, 2 * height / 3], axis=-1)
    return np.stack([v1, v2, v3], axis=1)


def _generate_long_triangle_vectorized(
    length: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Generates long, thin triangles for an array of lengths."""
    height = length / 8.0  # Make it very thin relative to its length
    v1 = np.stack([-length / 2, -height / 2], axis=-1)
    v2 = np.stack([-length / 2, height / 2], axis=-1)
    v3 = np.stack([length / 2, np.zeros_like(length)], axis=-1)
    return np.stack([v1, v2, v3], axis=1)


def _generate_random_vectorized(
    radius: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Generates random triangles for an array of radii."""
    angles = np.random.uniform(0, 2 * np.pi, (radius.shape[0], 3))
    radii = np.random.uniform(0, radius[:, np.newaxis], (radius.shape[0], 3))

    x_coords = radii * np.cos(angles)
    y_coords = radii * np.sin(angles)

    return np.stack([x_coords, y_coords], axis=-1)
