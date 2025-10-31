from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from sushi.backend.cpp import CPPBackend, CPPConfig
from sushi.backend.cuda import CUDABackend, CUDAConfig
from sushi.backend.numpy import NumpyBackend
from sushi.backend.opencv import OpenCVBackend
from sushi.backend.opengl import OpenGLBackend
from sushi.backend.pillow import PillowBackend
from sushi.golden_data import generate_triangles
from sushi.interface import Backend, Config, count_pixels_batch
from sushi.utils import np_image_loss

AllBackendsToTest = [
    (PillowBackend, None),
    (NumpyBackend, None),
    (OpenCVBackend, None),
    (OpenGLBackend, None),
    (CPPBackend, CPPConfig(method="scanline")),
    (CPPBackend, CPPConfig(method="pointwise")),
    (CUDABackend, CUDAConfig(method="naive-pixel-parallel")),
    (CUDABackend, CUDAConfig(method="naive-triangle-parallel")),
]


def maybe_skip_backend(
    backend: type[Backend], mode: Optional[Literal["draw", "drawloss"]] = None
) -> None:
    if not backend.is_available():
        pytest.skip(f"Backend {backend.name} is not available on this system.")
    elif mode is not None:
        res, err = backend.is_mode_supported(mode)
        if res == 0:
            pytest.skip(f"Backend {backend.name} does not support mode '{mode}'.")
        elif res == -1:
            pytest.skip(
                f"Backend {backend.name} cannot be tested "
                f"for mode '{mode}' due to an error: {err}"
            )


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


@pytest.fixture(scope="module")
def setup_data() -> TestImageData:
    """A pytest fixture to provide common test data."""
    width, height = 200, 150
    image_np = np.full((height, width, 3), 255, dtype=np.uint8)
    vertices = np.array([(10, 10), (195, 70), (50, 140)], dtype=np.int32)
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


@pytest.fixture(scope="module")
def reference_triangles_dataset() -> NDArray[np.int32]:
    """A pytest fixture to provide a very large set of test triangles for
    performance and scalability testing."""
    return generate_triangles(
        count=100,
        screen_width=256,
        screen_height=256,
        size="medium",
        shape_type="random",
        distribution="uniform",
        random_rotation=True,
        random_seed=38,
    )


@pytest.fixture(scope="module")
def reference_dataset_pixel_counts(
    reference_triangles_dataset: NDArray[np.int32],
) -> NDArray[np.int64]:
    """A pytest fixture to provide a reference pixel counts array for
    a very large set of test triangles. Calculated using the Numpy backend."""
    backend = NumpyBackend
    if not backend.is_available():
        pytest.skip(
            f"Reference backend {backend.name} is not available on this system."
        )

    pixel_counts = count_pixels_batch(
        vertices=reference_triangles_dataset,
        image_size=256,
        backend=backend,
    )

    return pixel_counts


@pytest.mark.parametrize(
    "raster_backend, config",
    [(b, c) for b, c in AllBackendsToTest if b != NumpyBackend],
)
def test_large_triangle_set_pixel_counts(
    reference_triangles_dataset: NDArray[np.int32],
    reference_dataset_pixel_counts: NDArray[np.int64],
    raster_backend: type["Backend"],
    config: Optional["Config"],
) -> None:
    """
    Tests that the pixel counting function produces correct counts for a large
    set of triangles by comparing to a reference dataset generated using the
    Numpy backend.
    """
    maybe_skip_backend(raster_backend, "drawloss")

    pixel_counts = count_pixels_batch(
        vertices=reference_triangles_dataset,
        image_size=256,
        backend=raster_backend,
        backend_config=config,
    )

    assert (
        pixel_counts.shape == reference_dataset_pixel_counts.shape
    ), "Pixel counts shape mismatch."

    # Define acceptance criteria:
    # At least 50% of triangle counts must match within a tolerance of 5 pixels or 40%.
    # At least 60% of triangle counts must match within a tolerance of 10 pixels or 60%.
    total_triangles = pixel_counts.shape[0]
    close_matches_fine = 0
    close_matches_coarse = 0
    close_match_fine_px_threshold = 5
    close_match_coarse_px_threshold = 10
    close_match_fine_pct_threshold = 0.4
    close_match_coarse_pct_threshold = 0.6
    for i in range(total_triangles):
        ref_count = reference_dataset_pixel_counts[i]
        test_count = pixel_counts[i]
        abs_diff = abs(int(test_count) - int(ref_count))
        pct_diff = abs_diff / float(ref_count) if ref_count > 0 else 0.0

        if (
            abs_diff <= close_match_fine_px_threshold
            or pct_diff <= close_match_fine_pct_threshold
        ):
            close_matches_fine += 1
        if (
            abs_diff <= close_match_coarse_px_threshold
            or pct_diff <= close_match_coarse_pct_threshold
        ):
            close_matches_coarse += 1

    match_rate_fine = close_matches_fine / total_triangles
    match_rate_coarse = close_matches_coarse / total_triangles
    assert match_rate_fine >= 0.50, (
        f"Only {match_rate_fine:.2%} of triangle pixel counts matched "
        f"within {close_match_fine_px_threshold} pixels or "
        f"{close_match_fine_pct_threshold * 100}% tolerance."
    )
    assert match_rate_coarse >= 0.60, (
        f"Only {match_rate_coarse:.2%} of triangle pixel counts matched "
        f"within {close_match_coarse_px_threshold} pixels or "
        f"{close_match_coarse_pct_threshold * 100}% tolerance."
    )

    print(
        f"Backend {raster_backend.name} pixel counting results: "
        f"{close_matches_fine} / {total_triangles} ({match_rate_fine:.2%}) matched "
        f"within {close_match_fine_px_threshold} pixels or "
        f"{close_match_fine_pct_threshold * 100}% tolerance; "
        f"{close_matches_coarse} / {total_triangles} ({match_rate_coarse:.2%}) matched "
        f"within {close_match_coarse_px_threshold} pixels or "
        f"{close_match_coarse_pct_threshold * 100}% tolerance."
    )


@pytest.mark.parametrize("raster_backend, config", AllBackendsToTest)
def test_can_clone_drawloss_context(
    raster_backend: type["Backend"], config: Optional["Config"]
) -> None:
    """
    Tests that the backend's drawloss context can be cloned without errors.
    """
    maybe_skip_backend(raster_backend, "drawloss")
    context = raster_backend.create_drawloss_context(
        background_image=np.zeros((10, 10, 3), dtype=np.uint8),
        target_image=np.zeros((10, 10, 3), dtype=np.uint8),
        config=config,
    )

    cloned_context = context.clone()
    assert cloned_context is not context, "Cloned context is the same instance."
    assert isinstance(
        cloned_context, type(context)
    ), "Cloned context is of incorrect type."


@pytest.mark.parametrize("raster_backend, config", AllBackendsToTest)
def test_can_clone_draw_context(
    raster_backend: type["Backend"], config: Optional["Config"]
) -> None:
    """
    Tests that the backend's draw context can be cloned without errors.
    """
    maybe_skip_backend(raster_backend, "draw")
    context = raster_backend.create_draw_context(
        background_image=np.zeros((10, 10, 3), dtype=np.uint8),
        config=config,
    )

    cloned_context = context.clone()
    assert cloned_context is not context, "Cloned context is the same instance."
    assert isinstance(
        cloned_context, type(context)
    ), "Cloned context is of incorrect type."

    cloned_context = context.clone()
    assert cloned_context is not context, "Cloned context is the same instance."
    assert isinstance(
        cloned_context, type(context)
    ), "Cloned context is of incorrect type."


@pytest.mark.parametrize("raster_backend, config", AllBackendsToTest)
def test_draw_single_match_reference(
    setup_data: TestImageData,
    raster_backend: type["Backend"],
    config: Optional["Config"],
) -> None:
    """
    Tests that drawing a triangle produces the exact expected image by
    comparing it to a pre-saved file. This verifies the drawing logic
    and coordinate orientation.
    """
    backend_name = raster_backend.name

    failed_image_path = Path(__file__).parent / f"data/failed_{backend_name}_draw.png"
    success_image_path = Path(__file__).parent / f"data/success_{backend_name}_draw.png"

    # Perform cleanup before skipping, so we don't leave old files around
    if success_image_path.exists():
        success_image_path.unlink()
    if failed_image_path.exists():
        failed_image_path.unlink()

    maybe_skip_backend(raster_backend, "draw")

    MATCH_PIXELS_THRESHOLD = 0.98  # 98% of pixels must match
    MATCH_PIXEL_VALUE_TOLERANCE = 2  # Allow small color differences due to rounding

    image_np = setup_data.image_np
    vertices = setup_data.vertices
    color = setup_data.color
    expected_image_np = setup_data.expected_image_np

    context = raster_backend.create_draw_context(
        background_image=image_np, config=config
    )

    drawn_image_np = context.draw(vertices, color)

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


@pytest.mark.parametrize("raster_backend, config", AllBackendsToTest)
def test_count_pixels_single_match_reference(
    setup_data: TestImageData,
    raster_backend: type["Backend"],
    config: Optional["Config"],
) -> None:
    """
    Tests that the pixel counting function produces a count close to the
    expected value. The expected value is derived from the saved reference image.
    """
    maybe_skip_backend(raster_backend, "drawloss")
    MATCH_RELATIVE_TOLERANCE = 0.025  # Allow 2.5% tolerance

    vertices = setup_data.vertices
    expected_pixel_count = int(
        (setup_data.expected_image_np.sum(axis=-1) != (3 * 255)).sum()
    )

    counted_pixels = int(
        count_pixels_batch(
            vertices=vertices[np.newaxis, ...],
            image_size=setup_data.width,
            backend=raster_backend,
            backend_config=config,
        ).sum()
    )

    print(
        f"Backend {raster_backend.name} counted {counted_pixels} pixels, "
        f"expected {expected_pixel_count}."
    )

    assert np.isclose(
        counted_pixels, expected_pixel_count, rtol=MATCH_RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize("raster_backend, config", AllBackendsToTest)
def test_drawloss_single(
    setup_data: TestImageData,
    raster_backend: type["Backend"],
    config: Optional["Config"],
) -> None:
    """
    Tests the draw-loss calculation by comparing the function's output
    to a manually calculated loss change. It also verifies that providing
    the optional `base_loss` gives the same result.
    """
    maybe_skip_backend(raster_backend, "drawloss")

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
        background_image=image_np, target_image=target_image_np, config=config
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
        f"{draw_loss:.0f}, expected {expected_loss_change:.0f}."
        f" (tolerance {tolerance_bounds[0]:.0f} to {tolerance_bounds[1]:.0f})"
    )

    np.testing.assert_allclose(
        draw_loss, expected_loss_change, rtol=MATCH_RELATIVE_TOLERANCE
    )
