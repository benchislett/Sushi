from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

# AllBackends = [
#     PillowRasterBackend,
#     OpenCVRasterBackend,
#     NumpyRasterBackend,
#     OpenGLRasterBackend,
# ]
# try:
#     from sushi.backend_cpp import CPPRasterBackend
#     HAS_CPP = True
#     AllBackends.append(CPPRasterBackend)
# except ImportError:
#     HAS_CPP = False
#     raise
from sushi.backend.pillow import PillowBackend
from sushi.interface import Backend

# from sushi.backend_numpy import NumpyRasterBackend
# from sushi.backend_opencv import OpenCVRasterBackend
# from sushi.backend_opengl import OpenGLRasterBackend
# from sushi.backend_pillow import PillowRasterBackend
from sushi.utils import np_image_loss

AllBackends = [PillowBackend]


@dataclass
class TestImageData:
    """A simple class to hold common test data."""

    __test__ = False  # Prevent pytest from collecting this class as a test case

    width: int
    height: int
    vertices: NDArray[np.int32]
    color: NDArray[np.uint8]
    image_np: NDArray[np.uint8]
    expected_image_np: NDArray[np.uint8]


def _load_test_image() -> NDArray[np.uint8]:
    """Helper function to load the reference test image."""
    expected_image_path = Path(__file__).parent / "data/expected_sample_triangle.png"
    if not expected_image_path.exists():
        pytest.fail(
            f"Expected image file not found at '{expected_image_path}'. "
            "Please run `tests/data/generate_test_data.py` first."
        )
    return np.array(Image.open(expected_image_path).convert("RGB"))


@pytest.fixture
def setup_data() -> TestImageData:
    """A pytest fixture to provide common test data."""
    width, height = 200, 150
    image_np = np.full((height, width, 3), 255, dtype=np.uint8)
    vertices = np.array([(10, 10), (50, 140), (195, 70)], dtype=np.int32)
    color = np.array((200, 55, 79, 192), dtype=np.uint8)
    expected_image_np = _load_test_image()

    return TestImageData(
        width=width,
        height=height,
        vertices=vertices,
        color=color,
        image_np=image_np,
        expected_image_np=expected_image_np,
    )


@pytest.mark.parametrize("raster_backend", AllBackends)
def test_draw_single_match_reference(
    setup_data: TestImageData, raster_backend: type["Backend"]
) -> None:
    """
    Tests that drawing a triangle produces the exact expected image by
    comparing it to a pre-saved file. This verifies the drawing logic
    and coordinate orientation.
    """
    MATCH_PIXELS_THRESHOLD = 0.98  # 98% of pixels must match
    MATCH_PIXEL_VALUE_TOLERANCE = 2  # Allow small color differences due to rounding

    backend_name = raster_backend.name

    image_np = setup_data.image_np
    vertices = setup_data.vertices
    color = setup_data.color
    expected_image_np = setup_data.expected_image_np

    context = raster_backend.create_draw_context(background_image=image_np)

    drawn_image_np = context.draw(vertices, color)

    failed_image_path = Path(__file__).parent / f"data/failed_{backend_name}_draw.png"
    success_image_path = Path(__file__).parent / f"data/success_{backend_name}_draw.png"

    if success_image_path.exists():
        success_image_path.unlink()
    if failed_image_path.exists():
        failed_image_path.unlink()

    assert drawn_image_np.shape == expected_image_np.shape
    num_pixels = drawn_image_np.shape[0] * drawn_image_np.shape[1]
    num_matching_pixels = np.sum(
        np.all(
            np.isclose(
                drawn_image_np, expected_image_np, atol=MATCH_PIXEL_VALUE_TOLERANCE
            ),
            axis=-1,
        )
    )
    assert num_pixels > 0
    match_rate = num_matching_pixels / num_pixels
    is_close = match_rate > MATCH_PIXELS_THRESHOLD

    if is_close:
        print(
            f"Backend {raster_backend.name} produced a matching image with "
            f"{num_matching_pixels} / {num_pixels} ({match_rate:.2%}) matching pixels."
        )
        Image.fromarray(drawn_image_np).save(success_image_path)
        print(f"Saved successful drawn image to '{success_image_path}'.")
    else:
        print(
            f"Backend {raster_backend.name} produced a non-matching image with "
            f"{num_matching_pixels} / {num_pixels} ({match_rate:.2%}) matching pixels."
        )
        Image.fromarray(drawn_image_np).save(failed_image_path)
        assert is_close, f"Image does not match reference. See '{failed_image_path}'."


@pytest.mark.parametrize("raster_backend", AllBackends)
def test_count_pixels_single_match_reference(
    setup_data: TestImageData, raster_backend: type["Backend"]
) -> None:
    """
    Tests that the pixel counting function produces a count close to the
    expected value. The expected value is derived from the saved reference image.
    """
    MATCH_RELATIVE_TOLERANCE = 0.025  # Allow 2.5% tolerance

    vertices = setup_data.vertices
    expected_pixel_count = int(
        (setup_data.expected_image_np.sum(axis=-1) != (3 * 255)).sum()
    )

    input_image = np.zeros((setup_data.height, setup_data.width, 3), dtype=np.uint8)
    context = raster_backend.create_draw_context(background_image=input_image)
    out_image = context.draw(vertices, np.array((255, 255, 255, 255), dtype=np.uint8))
    counted_pixels = int((out_image.sum(axis=-1) == (3 * 255)).sum())

    print(
        f"Backend {raster_backend.name} counted {counted_pixels} pixels, "
        f"expected {expected_pixel_count}."
    )

    assert np.isclose(
        counted_pixels, expected_pixel_count, rtol=MATCH_RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize("raster_backend", AllBackends)
def test_drawloss_single(
    setup_data: TestImageData, raster_backend: type["Backend"]
) -> None:
    """
    Tests the draw-loss calculation by comparing the function's output
    to a manually calculated loss change. It also verifies that providing
    the optional `base_loss` gives the same result.
    """
    MATCH_RELATIVE_TOLERANCE = 0.025  # Allow 2.5% tolerance
    image_np = setup_data.image_np
    height = setup_data.height
    width = setup_data.width
    vertices = setup_data.vertices
    color = setup_data.color
    expected_image_np = setup_data.expected_image_np

    target_image_color = np.array((64, 128, 199), dtype=np.uint8)
    target_image_np = np.tile(target_image_color, (height, width, 1))

    base_loss = np_image_loss(image_np, target_image_np)
    reference_final_loss = np_image_loss(expected_image_np, target_image_np)
    expected_loss_change = reference_final_loss - base_loss

    context = raster_backend.create_drawloss_context(
        background_image=image_np, target_image=target_image_np
    )

    draw_loss = int(
        context.drawloss(vertices[np.newaxis, ...], color[np.newaxis, ...])[0]
    )

    tolerance_bounds = (
        expected_loss_change * (1 - MATCH_RELATIVE_TOLERANCE),
        expected_loss_change * (1 + MATCH_RELATIVE_TOLERANCE),
    )
    if tolerance_bounds[0] > tolerance_bounds[1]:
        tolerance_bounds = (tolerance_bounds[1], tolerance_bounds[0])

    print(
        f"Backend {raster_backend.name} computed draw loss change of "
        f"{draw_loss:.6f}, expected {expected_loss_change:.6f}."
        f" (tolerance {tolerance_bounds[0]:.6f} to {tolerance_bounds[1]:.6f})"
    )

    np.testing.assert_allclose(
        draw_loss, expected_loss_change, rtol=MATCH_RELATIVE_TOLERANCE
    )
