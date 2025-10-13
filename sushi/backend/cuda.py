from typing import Any, ClassVar, Optional, override

import numpy as np
from numpy.typing import NDArray

from sushi.interface import (
    Backend,
    Config,
    DrawContext,
    DrawLossContext,
)
from sushi.utils import (
    check_image_rgb,
    check_triangle_vertices,
)

try:
    # This is the name of the module defined in the NB_MODULE macro
    from sushi_core import CUDABackend as CoreCUDABackend
except ImportError as e:
    raise ImportError(
        "CUDA backend not available. Make sure to build the C++/CUDA extension first."
    ) from e


class CUDAConfig(Config):
    """Configuration for the CUDA backend. (Currently empty)"""

    pass


class CUDADrawLossContext(DrawLossContext):
    """
    A drawloss context that uses the C++/CUDA backend.

    This class holds an instance of the C++ CUDABackend object, which manages
    all GPU memory and computation.
    """

    def __init__(
        self,
        *,
        background_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        config: CUDAConfig,
        **kwargs: Any,
    ) -> None:
        check_image_rgb(background_image)
        check_image_rgb(target_image)
        if background_image.shape != target_image.shape:
            raise ValueError(
                "Background and target images must have the same shape, "
                f"got {background_image.shape} and {target_image.shape}"
            )

        self.config = config
        # Instantiate the C++ backend. This call allocates memory on the GPU
        # and copies the image data over. The `self._core_backend` object
        # now explicitly holds the state of the CUDA data.
        self._core_backend = CoreCUDABackend(
            background_image=background_image, target_image=target_image
        )

    @override
    def clone(self) -> "CUDADrawLossContext":
        # To clone, we need to re-instantiate the C++ object with the same
        # initial data, as the internal state (GPU pointers) cannot be
        # directly copied from Python.
        # This assumes the original images are accessible or stored here if needed.
        # For this template, we show the principle but note that you might
        # need to store the initial numpy arrays if you want a true deep clone.
        # A simple re-creation from scratch is often sufficient.
        raise NotImplementedError(
            "Cloning requires re-initialization of the CUDA backend. "
            "Consider storing initial images if deep cloning is needed."
        )

    @override
    def drawloss(
        self, vertices: NDArray[np.int32], colors: NDArray[np.uint8]
    ) -> NDArray[np.int64]:
        """
        Calculates the rendering loss for a batch of triangles.
        """
        if vertices.shape[0] == 0:
            return np.array([], dtype=np.int64)
        check_triangle_vertices(vertices[0])
        if colors.ndim != 2 or colors.shape[1] != 4:
            raise ValueError(f"Colors must be a Nx4 array, got shape {colors.shape}")
        if vertices.shape[0] != colors.shape[0]:
            raise ValueError(
                "Vertices and colors must have the same batch size, got "
                f"{vertices.shape[0]} and {colors.shape[0]}"
            )

        # Create an output array for the C++ function to write into.
        out = np.empty((vertices.shape[0],), dtype=np.int64)

        # Call the C++ implementation. This transfers vertex/color data to the GPU,
        # runs the kernel, and transfers the results back.
        self._core_backend.drawloss(vertices, colors, out)

        return out


class CUDABackend(Backend):
    name: ClassVar[str] = "cuda"

    @classmethod
    def create_draw_context(cls: type["CUDABackend"], **kwargs: Any) -> DrawContext:
        raise NotImplementedError("CUDABackend is a drawloss-only backend.")

    @classmethod
    @override
    def create_drawloss_context(
        cls: type["CUDABackend"],
        *,
        background_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        config: Optional[Config] = None,
    ) -> DrawLossContext:
        if config is None:
            config = CUDAConfig()
        if not isinstance(config, CUDAConfig):
            raise TypeError(
                f"Config must be an instance of CUDAConfig, got {type(config)}"
            )
        return CUDADrawLossContext(
            background_image=background_image,
            target_image=target_image,
            config=config,
        )

    @classmethod
    @override
    def is_supported(cls: type["CUDABackend"]) -> bool:
        try:
            _ = cls.create_drawloss_context(
                background_image=np.zeros((10, 10, 3), dtype=np.uint8),
                target_image=np.zeros((10, 10, 3), dtype=np.uint8),
            )
            return True
        except Exception:
            return False
