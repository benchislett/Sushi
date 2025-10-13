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
    check_color_rgba,
    check_image_rgb,
    check_triangle_vertices,
)

try:
    from sushi_core import CPPRasterBackend as CoreBackend
except ImportError as e:
    raise ImportError(
        "C++ backend not available. Make sure to build the C++ extension first. "
        "Run 'pip install -e .' to build the extension."
    ) from e


class CPPConfig(Config):
    pass


class CPPDrawContext(DrawContext):
    """A drawing context that uses the C++ backend for rasterization."""

    def __init__(
        self,
        *,
        background_image: NDArray[np.uint8],
        config: CPPConfig,
    ) -> None:
        check_image_rgb(background_image)
        self.config = config
        self._image = background_image.copy()

    @override
    def clone(self) -> "CPPDrawContext":
        out_obj = self.__new__(self.__class__)
        out_obj.config = self.config
        out_obj._image = self._image.copy()
        return out_obj

    @override
    def get_image(self) -> NDArray[np.uint8]:
        return self._image

    @override
    def draw_inplace(
        self,
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        check_triangle_vertices(vertices)
        check_color_rgba(color)

        # Call the C++ implementation
        CoreBackend.triangle_draw_single_rgba_inplace(self._image, vertices, color)


class CPPDrawLossContext(DrawLossContext):
    """A drawloss context that uses the C++ backend for rasterization."""

    @override
    def __init__(
        self,
        *,
        background_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        config: CPPConfig,
        **kwargs: Any,
    ) -> None:
        check_image_rgb(background_image)
        check_image_rgb(target_image)
        if background_image.shape != target_image.shape:
            raise ValueError(
                "Background and target images must have the same shape, "
                f"got {background_image.shape} and {target_image.shape}"
            )
        self._background_image = background_image
        self._target_image = target_image
        self.config = config

    @override
    def clone(self) -> "CPPDrawLossContext":
        out_obj = self.__new__(self.__class__)
        out_obj._background_image = self._background_image.copy()
        out_obj._target_image = self._target_image.copy()
        out_obj.config = self.config
        return out_obj

    @override
    def drawloss(
        self, vertices: NDArray[np.int32], colors: NDArray[np.uint8]
    ) -> NDArray[np.int64]:
        # The C++ backend performs a direct, stateless calculation of the loss
        # for a batch of triangles without modifying a canvas.
        out = np.empty((vertices.shape[0],), dtype=np.int64)
        CoreBackend.triangle_drawloss_batch_rgba(
            self._background_image, self._target_image, vertices, colors, out
        )
        # Cast to int64 to match the interface spec, as loss is squared error.
        return out.astype(np.int64)


class CPPBackend(Backend):
    name: ClassVar[str] = "cpp"

    @classmethod
    @override
    def create_draw_context(
        cls: type["CPPBackend"],
        *,
        background_image: NDArray[np.uint8],
        config: Optional[Config] = None,
    ) -> DrawContext:
        if config is None:
            config = CPPConfig()
        if not isinstance(config, CPPConfig):
            raise TypeError(
                f"Config must be an instance of CPPConfig, got {type(config)}"
            )
        return CPPDrawContext(background_image=background_image, config=config)

    @classmethod
    @override
    def create_drawloss_context(
        cls: type["CPPBackend"],
        *,
        background_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        config: Optional[Config] = None,
    ) -> DrawLossContext:
        if config is None:
            config = CPPConfig()
        if not isinstance(config, CPPConfig):
            raise TypeError(
                f"Config must be an instance of CPPConfig, got {type(config)}"
            )
        return CPPDrawLossContext(
            background_image=background_image,
            target_image=target_image,
            config=config,
        )

    @classmethod
    @override
    def is_supported(cls: type["CPPBackend"]) -> bool:
        return True
