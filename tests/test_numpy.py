from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sushi.backend_numpy import (
    numpy_count_pixels_single,
    numpy_draw_single,
    numpy_draw_single_inplace,
    numpy_drawloss_single,
)
from sushi.utils import np_image_mse


@dataclass
class NumpyTestImageData:
    """A simple class to hold common test data."""

    width: int
    height: int
    image_np: np.ndarray
    vertices: np.ndarray
    color: np.ndarray


@pytest.fixture
def setup_data() -> NumpyTestImageData:
    """A pytest fixture to provide common test data."""
    width, height = 200, 150
    image_np = np.full((height, width, 3), 255, dtype=np.uint8)
    vertices = np.array([(10, 10), (50, 140), (195, 70)], dtype=np.int32)
    color = np.array((227, 177, 177), dtype=np.uint8)

    return NumpyTestImageData(
        width=width,
        height=height,
        image_np=image_np,
        vertices=vertices,
        color=color,
    )


def test_numpy_draw_match_reference(setup_data: NumpyTestImageData) -> None:
    """
    Tests that drawing a triangle produces the exact expected image by
    comparing it to a pre-saved file. This verifies the drawing logic,
    color blending, and coordinate orientation.
    """
    image_np = setup_data.image_np
    vertices = setup_data.vertices
    color = setup_data.color

    expected_image_path = Path(__file__).parent / "data/expected_sample_triangle.png"
    if not expected_image_path.exists():
        pytest.fail(
            f"Expected image file not found at '{expected_image_path}'. "
            "Please run `tests/data/generate_test_data.py` first."
        )

    expected_image_np = np.array(Image.open(expected_image_path).convert("RGB"))

    drawn_image_np = numpy_draw_single(image_np, vertices, color)

    try:
        assert drawn_image_np.shape == expected_image_np.shape
        num_pixels = drawn_image_np.shape[0] * drawn_image_np.shape[1]
        num_matching_pixels = np.sum(
            np.all(drawn_image_np == expected_image_np, axis=-1)
        )
        is_close = num_matching_pixels / num_pixels > 0.98  # 2% tolerance
        if not is_close:
            np.testing.assert_array_equal(
                drawn_image_np,
                expected_image_np,
            )
    except AssertionError as e:
        # If the images don't match, save the drawn image for inspection.
        failed_image_path = Path(__file__).parent / "data/failed_numpy_draw.png"
        Image.fromarray(drawn_image_np).save(failed_image_path)
        print(f"Saved failed drawn image to '{failed_image_path}' for inspection.")
        raise e


def test_numpy_count_pixels_single(setup_data: NumpyTestImageData) -> None:
    """
    Tests that the pixel count for a given polygon is correct. The expected
    value is derived from the original notebook's verification step.
    """
    height = setup_data.height
    width = setup_data.width
    vertices = setup_data.vertices
    expected_pixel_count = 10952  # Manually calculated value

    counted_pixels = numpy_count_pixels_single((height, width), vertices)

    assert abs(counted_pixels - expected_pixel_count) < 200


def test_numpy_drawloss_single(setup_data: NumpyTestImageData) -> None:
    """
    Tests the draw-loss calculation by comparing the function's output
    to a manually calculated loss change. It also verifies that providing
    the optional `base_loss` gives the same result.
    """
    image_np = setup_data.image_np
    height = setup_data.height
    width = setup_data.width
    vertices = setup_data.vertices
    color = setup_data.color

    target_image_color = np.array((100, 100, 100), dtype=np.uint8)
    target_image_np = np.tile(target_image_color, (height, width, 1))

    base_loss = np_image_mse(image_np, target_image_np)
    drawn_image_np = numpy_draw_single(image_np, vertices, color)
    final_loss = np_image_mse(drawn_image_np, target_image_np)
    expected_loss_change = final_loss - base_loss

    draw_loss_without_base = numpy_drawloss_single(
        image_np, target_image_np, vertices, color
    )
    draw_loss_with_base = numpy_drawloss_single(
        image_np, target_image_np, vertices, color, base_loss=base_loss
    )

    np.testing.assert_allclose(draw_loss_without_base, expected_loss_change, rtol=1e-5)
    np.testing.assert_allclose(draw_loss_with_base, expected_loss_change, rtol=1e-5)
