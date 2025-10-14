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
from typing import Literal, Optional

import numpy as np

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
}

ShapeType = Literal[
    "equilateral", "right", "random", "wide", "tall", "slanted-up", "slanted-down"
]
SizeType = Literal["micro", "tiny", "small", "medium", "large", "very_large", "huge"]
DistributionType = Literal["uniform", "normal", "spaced"]

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
) -> np.ndarray:
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
    elif distribution == "spaced":
        centers = _generate_spaced_centers(count, screen_width, screen_height)
    else:
        raise ValueError(
            f"Invalid distribution '{distribution}'. "
            "Choose 'uniform', 'normal', or 'spaced'."
        )

    # 2. Generate each triangle based on shape, size, and center point
    triangles = []
    min_len, max_len = SIZE_RANGES[size]

    for i in range(count):
        length = np.random.uniform(min_len, max_len)
        center = centers[i]
        angle = 0

        # Select the appropriate shape generation function

        # Triangles that can be fully randomly rotated
        if shape_type in ["equilateral", "right", "random"]:
            if shape_type == "equilateral":
                base_tri = _generate_equilateral(length)
            elif shape_type == "right":
                base_tri = _generate_right(length, length)  # Isosceles
            elif shape_type == "random":
                base_tri = _generate_random(length / 2)  # Use length as a radius

            # Apply a full random rotation for these types if enabled
            if random_rotation:
                angle = np.random.uniform(0, 2 * np.pi)

        # Triangles with constrained orientations that get slight jitter and flipping
        elif shape_type in ["wide", "tall", "slanted-up", "slanted-down"]:
            base_tri = _generate_long_triangle(length)

            if random_rotation:
                # Randomly flip the triangle along its length
                if np.random.rand() < 0.5:
                    base_tri[:, 0] *= -1

                # Add small random rotation (jitter)
                jitter = np.random.uniform(-0.1 * np.pi, 0.1 * np.pi)
            else:
                jitter = 0

            if shape_type == "wide":
                angle = 0 + jitter
            elif shape_type == "tall":
                angle = np.pi / 2 + jitter
            elif shape_type == "slanted-up":
                angle = -np.pi / 4 + jitter  # y = -x diagonal
            elif shape_type == "slanted-down":
                angle = np.pi / 4 + jitter  # y = x diagonal
        else:
            raise ValueError(f"Invalid shape_type '{shape_type}'.")

        # Apply rotation if any
        if angle != 0:
            rotation_matrix = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            base_tri = np.dot(base_tri, rotation_matrix.T)

        # Translate the generated triangle (centered at origin) to its final position
        translated_tri = base_tri + center
        triangles.append(translated_tri)

    return np.array(triangles, dtype=np.int32)


# --- Helper Functions for Distributions ---


def _generate_uniform_centers(count: int, width: int, height: int) -> np.ndarray:
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
) -> np.ndarray:
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


def _generate_spaced_centers(count: int, width: int, height: int) -> np.ndarray:
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


def _generate_equilateral(side: float) -> np.ndarray:
    """Generates an equilateral triangle centered at the origin."""
    height = side * np.sqrt(3) / 2
    # Vertices are calculated based on geometric properties
    v1 = np.array([0, 2 * height / 3])
    v2 = np.array([-side / 2, -height / 3])
    v3 = np.array([side / 2, -height / 3])
    return np.array([v1, v2, v3])


def _generate_right(width: float, height: float) -> np.ndarray:
    """Generates a right triangle centered at the origin."""
    # Create vertices for a right triangle at (0,0), (width,0), (0,height)
    # and then translate them so the centroid is at the origin.
    v1 = np.array([-width / 3, -height / 3])
    v2 = np.array([2 * width / 3, -height / 3])
    v3 = np.array([-width / 3, 2 * height / 3])
    return np.array([v1, v2, v3])


def _generate_long_triangle(length: float) -> np.ndarray:
    """Generates a long, thin triangle for wide/tall shapes."""
    height = length / 8.0  # Make it very thin relative to its length
    # Shape has two vertices on one side and one on the other
    return np.array(
        [[-length / 2, -height / 2], [-length / 2, height / 2], [length / 2, 0]]
    )


def _generate_random(radius: float) -> np.ndarray:
    """Generates a random triangle with vertices within a given radius."""
    angles = np.random.uniform(0, 2 * np.pi, 3)
    radii = np.random.uniform(0, radius, 3)

    x_coords = radii * np.cos(angles)
    y_coords = radii * np.sin(angles)

    return np.vstack([x_coords, y_coords]).T


# --- Example Usage & Visualization ---

if __name__ == "__main__":
    # Dynamically import matplotlib only when running as a script
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    def plot_triangles_on_ax(
        ax: plt.Axes,
        triangles: np.ndarray,
        screen_width: int,
        screen_height: int,
        title: str,
    ) -> None:
        """
        Draws the generated triangles onto a given Matplotlib Axes object.
        """
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, screen_width)
        ax.set_ylim(0, screen_height)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()  # Invert y-axis to match screen coordinates (0,0 at top-left)

        for tri in triangles:
            polygon = Polygon(tri, edgecolor="blue", facecolor="lightblue", alpha=0.5)
            ax.add_patch(polygon)

        ax.grid(True)

    # Configuration
    SCREEN_WIDTH = 512
    SCREEN_HEIGHT = 512
    RANDOM_SEED = 42
    OUTPUT_FILENAME = "triangle_examples.png"

    print("Generating example triangle sets...")

    # Define all example configurations
    examples = [
        {
            "title": "Medium, Rotated Right Triangles (Spaced)",
            "params": {
                "count": 100,
                "screen_width": SCREEN_WIDTH,
                "screen_height": SCREEN_HEIGHT,
                "shape_type": "right",
                "size": "medium",
                "distribution": "spaced",
                "random_seed": RANDOM_SEED,
            },
        },
        {
            "title": "Small, Equilateral (Custom Normal Dist.)",
            "params": {
                "count": 300,
                "screen_width": SCREEN_WIDTH,
                "screen_height": SCREEN_HEIGHT,
                "shape_type": "equilateral",
                "size": "small",
                "distribution": "normal",
                "distribution_mean": (SCREEN_WIDTH * 0.2, SCREEN_HEIGHT * 0.2),
                "distribution_std_dev": (SCREEN_WIDTH / 15, SCREEN_HEIGHT / 15),
                "random_seed": RANDOM_SEED,
            },
        },
        {
            "title": "Large, Wide (Uniform, No Jitter/Flip)",
            "params": {
                "count": 300,
                "screen_width": SCREEN_WIDTH,
                "screen_height": SCREEN_HEIGHT,
                "shape_type": "wide",
                "size": "large",
                "distribution": "uniform",
                "random_rotation": False,
                "random_seed": RANDOM_SEED,
            },
        },
        {
            "title": "Medium, Slanted-Up (With Jitter/Flip)",
            "params": {
                "count": 300,
                "screen_width": SCREEN_WIDTH,
                "screen_height": SCREEN_HEIGHT,
                "shape_type": "slanted-up",
                "size": "medium",
                "distribution": "uniform",
                "random_rotation": True,
                "random_seed": RANDOM_SEED,
            },
        },
        {
            "title": "Medium, Tall (Spaced, With Jitter/Flip)",
            "params": {
                "count": 200,
                "screen_width": SCREEN_WIDTH,
                "screen_height": SCREEN_HEIGHT,
                "shape_type": "tall",
                "size": "medium",
                "distribution": "spaced",
                "random_rotation": True,
                "random_seed": RANDOM_SEED,
            },
        },
        {
            "title": "Huge, Random (Normal Distribution)",
            "params": {
                "count": 25,
                "screen_width": SCREEN_WIDTH,
                "screen_height": SCREEN_HEIGHT,
                "shape_type": "random",
                "size": "huge",
                "distribution": "normal",
                "random_seed": RANDOM_SEED,
            },
        },
        {
            "title": "Large, Random (Centered Distribution)",
            "params": {
                "count": 50,
                "screen_width": SCREEN_WIDTH,
                "screen_height": SCREEN_HEIGHT,
                "shape_type": "random",
                "size": "large",
                "distribution": "normal",
                "distribution_mean": (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2),
                "distribution_std_dev": (0, 0),
                "random_seed": RANDOM_SEED,
            },
        },
        {
            "title": "Micro, Equilateral (Spaced Distribution) With Jitter",
            "params": {
                "count": 1000,
                "screen_width": SCREEN_WIDTH,
                "screen_height": SCREEN_HEIGHT,
                "shape_type": "equilateral",
                "size": "micro",
                "distribution": "spaced",
                "random_rotation": True,
                "random_seed": RANDOM_SEED,
            },
        },
        {
            "title": "Tiny, Right Triangles (Uniform Distribution)",
            "params": {
                "count": 1000,
                "screen_width": SCREEN_WIDTH,
                "screen_height": SCREEN_HEIGHT,
                "shape_type": "right",
                "size": "tiny",
                "distribution": "uniform",
                "random_seed": RANDOM_SEED,
                "random_rotation": False,
            },
        },
    ]

    # Create a grid of plots.
    num_examples = len(examples)
    grid_cols = 3
    grid_rows = math.ceil(num_examples / grid_cols)
    fig, axes = plt.subplots(
        grid_rows, grid_cols, figsize=(5 * grid_cols, 5 * grid_rows)
    )
    axes = axes.flatten()

    # Generate and plot each example
    for i, example in enumerate(examples):
        assert isinstance(example, dict)
        assert isinstance(example["params"], dict)
        assert isinstance(example["title"], str)
        triangles = generate_triangles(**example["params"])
        plot_triangles_on_ax(
            ax=axes[i],
            triangles=triangles,
            screen_width=SCREEN_WIDTH,
            screen_height=SCREEN_HEIGHT,
            title=example["title"],
        )

    # Hide any unused subplots
    for i in range(num_examples, len(axes)):
        axes[i].axis("off")

    # Adjust layout and save the final image
    fig.suptitle("Triangle Generation Examples", fontsize=24)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(OUTPUT_FILENAME, dpi=400, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory

    print(f"Stitched plot saved to '{OUTPUT_FILENAME}'")
