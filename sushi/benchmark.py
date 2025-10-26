#!/usr/bin/env python3

import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import rich.box
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from numpy.typing import NDArray
from rich.console import Console
from rich.live import Live
from rich.table import Table
from tqdm import tqdm

# --- Import All Available Backends ---
# Add or remove backends as needed for your environment.
from sushi.backend.cpp import CPPBackend, CPPConfig
from sushi.backend.cuda import CUDABackend, CUDAConfig
from sushi.backend.numpy import NumpyBackend
from sushi.backend.opencv import OpenCVBackend
from sushi.backend.opengl import OpenGLBackend
from sushi.backend.pillow import PillowBackend
from sushi.golden_data import generate_triangles

# Assuming the provided interface and golden data are in these locations
from sushi.interface import Backend, Config, DrawLossContext


@dataclass(frozen=True)
class BenchmarkConfig:
    backend: Type[Backend]
    description: str
    batch_sizes: List[int]
    config: Optional[Config] = None
    short_name: str = ""

    def __post_init__(self) -> None:
        if not self.short_name:
            object.__setattr__(self, "short_name", self.description)


@dataclass(frozen=True)
class BenchmarkInputData:
    vertices: NDArray[np.int32]
    colors: NDArray[np.uint8]
    config_name: str
    config: dict[str, Any]


# Default benchmark batch size: only one test for backends that run sequentially
BENCHMARK_DEFAULT_BATCH_SIZES = [1]
# Larger batch size for C++ backends that can amortize the binding overhead
BENCHMARK_CPP_BATCH_SIZES = [100]
# Very large batch sizes for CUDA backends that leverage massive parallelism
BENCHMARK_CUDA_BATCH_SIZES = [10, 100, 1000, 10_000, 100_000]


ALL_BENCHMARK_CONFIGS: List[BenchmarkConfig] = [
    # Python Reference Backends
    BenchmarkConfig(
        backend=NumpyBackend,
        description="NumPy",
        batch_sizes=BENCHMARK_DEFAULT_BATCH_SIZES,
    ),
    BenchmarkConfig(
        backend=OpenCVBackend,
        description="OpenCV",
        batch_sizes=BENCHMARK_DEFAULT_BATCH_SIZES,
    ),
    BenchmarkConfig(
        backend=PillowBackend,
        description="Pillow",
        batch_sizes=BENCHMARK_DEFAULT_BATCH_SIZES,
    ),
    BenchmarkConfig(
        backend=OpenGLBackend,
        description="OpenGL",
        batch_sizes=BENCHMARK_DEFAULT_BATCH_SIZES,
    ),
    # C++ Backends
    BenchmarkConfig(
        backend=CPPBackend,
        config=CPPConfig(method="scanline"),
        description="C++ Scanline",
        batch_sizes=BENCHMARK_CPP_BATCH_SIZES,
    ),
    BenchmarkConfig(
        backend=CPPBackend,
        config=CPPConfig(method="pointwise"),
        description="C++ Pointwise",
        batch_sizes=BENCHMARK_CPP_BATCH_SIZES,
    ),
    # CUDA Backends
    BenchmarkConfig(
        backend=CUDABackend,
        config=CUDAConfig(method="naive-triangle-parallel"),
        description="CUDA Naive Triangle-Per-Thread",
        short_name="CUDA Trianglewise",
        batch_sizes=BENCHMARK_CUDA_BATCH_SIZES,
    ),
    BenchmarkConfig(
        backend=CUDABackend,
        config=CUDAConfig(method="naive-pixel-parallel"),
        description="CUDA Naive Pixel-Per-Thread",
        short_name="CUDA Pixelwise",
        batch_sizes=BENCHMARK_CUDA_BATCH_SIZES,
    ),
]


def plot_triangles_on_ax(
    ax: Axes,
    triangles: NDArray[np.int32],
    screen_width: int,
    screen_height: int,
    title: str,
) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, screen_width)
    ax.set_ylim(0, screen_height)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()  # Invert y-axis to match screen coordinates (0,0 at top-left)

    for tri in triangles:
        polygon = Polygon(tri, edgecolor="blue", facecolor="lightblue", alpha=0.5)
        ax.add_patch(polygon)

    ax.grid(True)


def plot_example_triangles(
    benchmark_configs: dict[str, Any],
    out_filename: str = "benchmark_triangle_samples.png",
) -> None:
    print("Generating benchmark visualization...")

    print("Setting up plot grid...")

    # Create a grid of plots.
    num_examples = len(benchmark_configs)
    grid_cols = 3
    grid_rows = math.ceil(num_examples / grid_cols)
    fig, axes = plt.subplots(
        grid_rows, grid_cols, figsize=(5 * grid_cols, 5 * grid_rows)
    )
    axes = axes.flatten()

    # Generate and plot each example
    for i, (title, example) in enumerate(benchmark_configs.items()):
        assert isinstance(example, dict)
        image_size = example["screen_width"]
        triangles = generate_triangles(**example)
        plot_triangles_on_ax(
            ax=axes[i],
            triangles=triangles,
            screen_width=image_size,
            screen_height=image_size,
            title=title,
        )

    # Hide any unused subplots
    for i in range(num_examples, len(axes)):
        axes[i].axis("off")

    print("Saving stitched plot...")

    # Adjust layout and save the final image
    fig.suptitle("Triangle Generation Examples", fontsize=24)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.savefig(out_filename, dpi=400, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory

    print(f"Stitched plot saved to '{out_filename}'")


# --- Golden Data for Testing and Benchmarking ---
def get_golden_benchmark_data(N: int) -> dict[str, dict[str, Any]]:
    """
    Returns a dictionary of golden benchmark datasets for testing and benchmarking.
    Each key is a descriptive name, and the value is a configuration dictionary
    that can be passed to `generate_triangles()`.
    """
    DEFAULT_IMAGE_SIZE = 256
    LARGE_IMAGE_SIZE = 512
    VERY_LARGE_IMAGE_SIZE = 1024
    RANDOM_SEED = 42
    benchmark_configs = {
        "Very Large Random on 256x256": {
            "count": N,
            "screen_width": DEFAULT_IMAGE_SIZE,
            "screen_height": DEFAULT_IMAGE_SIZE,
            "shape_type": "random",
            "size": "very_large",
            "distribution": "center",
            "random_rotation": True,
            "random_seed": RANDOM_SEED,
            "short_name": "XL Random",
        },
        "Tiny Spaced on 256x256": {
            "count": N,
            "screen_width": DEFAULT_IMAGE_SIZE,
            "screen_height": DEFAULT_IMAGE_SIZE,
            "shape_type": "equilateral",
            "size": "tiny",
            "distribution": "spaced",
            "random_rotation": True,
            "random_seed": RANDOM_SEED,
            "short_name": "Tiny Spaced",
        },
        "Full Coverage on 256x256": {
            "count": N,
            "screen_width": DEFAULT_IMAGE_SIZE,
            "screen_height": DEFAULT_IMAGE_SIZE,
            "shape_type": "equilateral",
            "size": "massive",
            "distribution": "center",
            "random_rotation": True,
            "random_seed": RANDOM_SEED,
            "short_name": "Full Coverage",
        },
        "Small Centered Random on 256x256": {
            "count": N,
            "screen_width": DEFAULT_IMAGE_SIZE,
            "screen_height": DEFAULT_IMAGE_SIZE,
            "shape_type": "random",
            "size": "small",
            "distribution": "center",
            "random_rotation": True,
            "random_seed": RANDOM_SEED,
            "short_name": "Small Centered",
        },
        "Concentrated Micro on 256x256": {
            "count": N,
            "screen_width": DEFAULT_IMAGE_SIZE,
            "screen_height": DEFAULT_IMAGE_SIZE,
            "shape_type": "equilateral",
            "size": "micro",
            "distribution": "normal",
            "distribution_mean": (DEFAULT_IMAGE_SIZE / 2, DEFAULT_IMAGE_SIZE / 2),
            "distribution_std_dev": (DEFAULT_IMAGE_SIZE / 20, DEFAULT_IMAGE_SIZE / 20),
            "random_rotation": True,
            "random_seed": RANDOM_SEED,
            "short_name": "Concentrated Micro",
        },
        "Large Slanted on 256x256": {
            "count": N,
            "screen_width": DEFAULT_IMAGE_SIZE,
            "screen_height": DEFAULT_IMAGE_SIZE,
            "shape_type": "slanted-down",
            "size": "large",
            "distribution": "uniform",
            "random_rotation": True,
            "random_seed": RANDOM_SEED,
            "short_name": "Lg Slanted",
        },
        "Large Uniform on 512x512": {
            "count": N,
            "screen_width": LARGE_IMAGE_SIZE,
            "screen_height": LARGE_IMAGE_SIZE,
            "shape_type": "equilateral",
            "size": "large",
            "distribution": "uniform",
            "random_rotation": True,
            "random_seed": RANDOM_SEED,
            "short_name": "Lg Uniform Lg Canvas",
        },
        "Very Large Uniform on 1024x1024": {
            "count": N,
            "screen_width": VERY_LARGE_IMAGE_SIZE,
            "screen_height": VERY_LARGE_IMAGE_SIZE,
            "shape_type": "equilateral",
            "size": "very_large",
            "distribution": "uniform",
            "random_rotation": True,
            "random_seed": RANDOM_SEED,
            "short_name": "XL Uniform XL Canvas",
        },
        "Sparse Small Random on 1024x1024": {
            "count": N,
            "screen_width": VERY_LARGE_IMAGE_SIZE,
            "screen_height": VERY_LARGE_IMAGE_SIZE,
            "shape_type": "random",
            "size": "small",
            "distribution": "spaced",
            "random_rotation": True,
            "random_seed": RANDOM_SEED,
            "short_name": "Sm Sparse XL Canvas",
        },
    }
    return benchmark_configs


# --- Benchmark Configuration ---

WARMUP_RUNS = (
    0  # Number of initial runs to discard for JIT compilation, cache warming, etc.
)
TIMED_RUNS = 1  # Number of runs to average for the final timing.


@dataclass
class BenchmarkResult:
    backend: str
    scenario: str

    tris_per_sec: float
    pixels_per_sec: float


def count_pixels_in_batch(
    vertices: NDArray[np.int32],
    image_size: int,
    backend: type["Backend"],
    backend_config: Optional[Config],
) -> int:
    """
    Calculates the total number pixels that are covered by a DrawLoss batch.

    Note: Triangles are treated as independent. There are no coverage assumptions
    or culling rules. The return value is the sum of the pixel counts of each triangle.
    """
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    sample_color = np.array([255, 255, 255, 255], dtype=np.uint8)

    context = backend.create_drawloss_context(
        background_image=canvas, target_image=canvas, config=backend_config
    )

    result = context.drawloss(vertices, np.tile(sample_color, (vertices.shape[0], 1)))
    return int((result // (255 * 255 * 3)).sum())


def run_benchmark(
    input_data: BenchmarkInputData,
    run_config: BenchmarkConfig,
) -> Optional[BenchmarkResult]:
    """
    Runs a single benchmark configuration for a given backend and returns the result.
    """
    vertices, colors, config = (
        input_data.vertices,
        input_data.colors,
        input_data.config,
    )
    backend, backend_config = run_config.backend, run_config.config
    support_level, _ = backend.is_mode_supported("drawloss")
    if support_level <= 0:
        return None

    assert (
        config["screen_width"] == config["screen_height"]
    ), "Currently only square images are supported in benchmarks."
    size = config["screen_width"]
    background_image = np.random.randint(2, 253, (size, size, 3), dtype=np.uint8)
    target_image = np.random.randint(2, 253, (size, size, 3), dtype=np.uint8)

    N = config["count"]

    total_pixels = count_pixels_in_batch(vertices, size, backend, backend_config)

    context = backend.create_drawloss_context(
        background_image=background_image,
        target_image=target_image,
        config=backend_config,
    )

    for _ in range(WARMUP_RUNS):
        _ = context.drawloss(vertices, colors)

    timings = []
    for _ in range(TIMED_RUNS):
        start_time = time.perf_counter()
        _ = context.drawloss(vertices, colors)
        end_time = time.perf_counter()
        timings.append(end_time - start_time)

    avg_time = sum(timings) / len(timings)
    tris_per_sec = N / avg_time
    pixels_per_sec = total_pixels / avg_time

    return BenchmarkResult(
        backend=run_config.short_name or "",
        scenario=config["short_name"],
        tris_per_sec=tris_per_sec,
        pixels_per_sec=pixels_per_sec,
    )


def main() -> None:
    """
    Main function to generate previews and execute all benchmarks.
    """
    console = Console()

    # 2. Run Benchmarks
    console.print("Starting `drawloss` performance benchmarks...")

    # Visualize the benchmark scenarios for reference
    sample_benchmark_data = get_golden_benchmark_data(N=100)
    plot_example_triangles(sample_benchmark_data)

    results_data: dict[str, dict[str, Any]] = defaultdict(dict)

    scenario_names = [v["short_name"] for v in sample_benchmark_data.values()]

    available_backend_configs = []
    for config in ALL_BENCHMARK_CONFIGS:
        if config.backend.is_available():
            available_backend_configs.append(config)
        else:
            console.print(
                f"[yellow]Skipping {config.description}: "
                "backend not available.[/yellow]"
            )

    console.print("")

    def generate_table(current_results: Any) -> Table:
        """Generates a Rich Table from the current benchmark results."""
        table = Table(
            title="Benchmark Results (Triangles/sec | Pixels/sec)",
            box=rich.box.ROUNDED,
            caption="[dim]T/s = Triangles per Second | P/s = Pixels per Second[/dim]",
        )
        table.add_column(
            "Benchmark Scenario",
            style="cyan",
            no_wrap=True,
        )
        available_backend_names = [b.short_name for b in available_backend_configs]
        for backend_name in available_backend_names:
            table.add_column(
                backend_name,
                justify="right",
            )

        for scenario in scenario_names:
            row_cells = [scenario]
            for backend in available_backend_names:
                result: Optional[BenchmarkResult] = current_results[scenario].get(
                    backend
                )

                if result:
                    unit_scales = [" ", "k", "M", "G", "T"]
                    tris_per_sec = result.tris_per_sec
                    scale_index = 0
                    while tris_per_sec >= 1000 and scale_index < len(unit_scales) - 1:
                        tris_per_sec /= 1000
                        scale_index += 1
                    tris_per_sec_str = (
                        f"{tris_per_sec:.1f}"
                        if scale_index > 0
                        else f"{tris_per_sec:.0f}"
                    )
                    tris_per_sec_str = (
                        f"{tris_per_sec_str} "
                        f"[green]{unit_scales[scale_index]}T/s[/green]"
                    )
                    pixels_per_sec = result.pixels_per_sec
                    scale_index = 0
                    while pixels_per_sec >= 1000 and scale_index < len(unit_scales) - 1:
                        pixels_per_sec /= 1000
                        scale_index += 1
                    pixels_per_sec_str = (
                        f"{pixels_per_sec:.1f}"
                        if scale_index > 0
                        else f"{pixels_per_sec:.0f}"
                    )
                    pixels_per_sec_str = (
                        f"[dim]{pixels_per_sec_str} {unit_scales[scale_index]}P/s[/dim]"
                    )
                    cell_text = f"{tris_per_sec_str}\n{pixels_per_sec_str}"
                    row_cells.append(cell_text)
                else:
                    # Placeholder for pending benchmarks
                    row_cells.append("[gray]Pending...[/gray]")
            table.add_row(*row_cells)
        return table

    # Group benchmark configs by batch size to minimize redundant data generation
    all_batch_sizes_to_run = set(
        bs for config in available_backend_configs for bs in config.batch_sizes
    )
    with Live(
        generate_table(results_data),
        refresh_per_second=4,
        console=console,
        vertical_overflow="visible",
    ) as live:
        for batch_size in sorted(all_batch_sizes_to_run):
            benchmark_input_configs = get_golden_benchmark_data(N=batch_size)
            benchmark_input_data: List[BenchmarkInputData] = []
            for config_name, config in benchmark_input_configs.items():  # type: ignore
                assert isinstance(config, dict)
                vertices = generate_triangles(**config)
                colors = np.random.randint(1, 254, (batch_size, 4), dtype=np.uint8)
                benchmark_input_data.append(
                    BenchmarkInputData(
                        vertices=vertices,
                        colors=colors,
                        config_name=config_name,
                        config=config,
                    )
                )
            for benchmark_config in available_backend_configs:
                if batch_size not in benchmark_config.batch_sizes:
                    continue

                for input_elem in benchmark_input_data:
                    result = run_benchmark(
                        input_data=input_elem,
                        run_config=benchmark_config,
                    )
                    if result is not None:
                        results_data[result.scenario][result.backend] = result
                        live.update(generate_table(results_data))

    print()
    print("Benchmark run complete.")


if __name__ == "__main__":
    main()
