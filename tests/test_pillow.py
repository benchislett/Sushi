from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sushi.backend_pillow import (
    pillow_count_pixels_single,
    pillow_draw_single,
    pillow_drawloss_single,
)
from sushi.utils import np_image_mse


@pytest.fixture
def setup_data() -> dict:
    """A pytest fixture to provide common test data."""
    width, height = 200, 150
    image_np = np.full((height, width, 3), 255, dtype=np.uint8)
    vertices = np.array([(10, 10), (50, 140), (195, 70)], dtype=np.int32)
    color = np.array((200, 100, 100, 128), dtype=np.uint8)
    return {
        "width": width,
        "height": height,
        "image_np": image_np,
        "vertices": vertices,
        "color": color,
    }


def test_pillow_draw_match_reference(setup_data: dict) -> None:
    """
    Tests that drawing a triangle produces the exact expected image by
    comparing it to a pre-saved file. This verifies the drawing logic,
    color blending, and coordinate orientation.
    """
    image_np = setup_data["image_np"]
    vertices = setup_data["vertices"]
    color = setup_data["color"]

    expected_image_path = Path(__file__).parent / "data/expected_sample_triangle.png"
    if not expected_image_path.exists():
        pytest.fail(
            f"Expected image file not found at '{expected_image_path}'. "
            "Please run `tests/data/generate_test_data.py` first."
        )

    expected_image_np = np.array(Image.open(expected_image_path).convert("RGB"))

    drawn_image_np = pillow_draw_single(image_np, vertices, color)

    np.testing.assert_array_equal(drawn_image_np, expected_image_np)


def test_pillow_count_pixels_single(setup_data: dict) -> None:
    """
    Tests that the pixel count for a given polygon is correct. The expected
    value is derived from the original notebook's verification step.
    """
    height = setup_data["height"]
    width = setup_data["width"]
    vertices = setup_data["vertices"]
    expected_pixel_count = 10952  # Manually calculated value

    counted_pixels = pillow_count_pixels_single((height, width), vertices)

    assert counted_pixels == expected_pixel_count


def test_pillow_drawloss_single(setup_data: dict) -> None:
    """
    Tests the draw-loss calculation by comparing the function's output
    to a manually calculated loss change. It also verifies that providing
    the optional `base_loss` gives the same result.
    """
    image_np = setup_data["image_np"]
    height = setup_data["height"]
    width = setup_data["width"]
    vertices = setup_data["vertices"]
    color = setup_data["color"]

    target_image_color = np.array((100, 100, 100), dtype=np.uint8)
    target_image_np = np.tile(target_image_color, (height, width, 1))

    base_loss = np_image_mse(image_np, target_image_np)
    drawn_image_np = pillow_draw_single(image_np, vertices, color)
    final_loss = np_image_mse(drawn_image_np, target_image_np)
    expected_loss_change = final_loss - base_loss

    draw_loss_without_base = pillow_drawloss_single(
        image_np, target_image_np, vertices, color
    )
    draw_loss_with_base = pillow_drawloss_single(
        image_np, target_image_np, vertices, color, base_loss=base_loss
    )

    np.testing.assert_allclose(draw_loss_without_base, expected_loss_change, rtol=1e-5)
    np.testing.assert_allclose(draw_loss_with_base, expected_loss_change, rtol=1e-5)
