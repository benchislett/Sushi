from dataclasses import dataclass
from typing import Any, ClassVar, Optional, override

import cv2
import numpy as np
from numpy.typing import NDArray

from sushi.interface import (
    Backend,
    Config,
    DrawContext,
    DrawLossContext,
    calculate_drawloss_using_draw_context,
)
from sushi.utils import (
    check_color_rgba,
    check_image_rgb,
    check_triangle_vertices,
)


class OpenCVConfig(Config):
    """Configuration for the OpenCV backend."""

    pass


class OpenCVDrawContext(DrawContext):
    """A drawing context that uses OpenCV for rasterization."""

    def __init__(
        self,
        *,
        background_image: NDArray[np.uint8],
        config: OpenCVConfig,
    ) -> None:
        check_image_rgb(background_image)
        self.config = config
        self._image = background_image.copy()

    @override
    def clone(self) -> "OpenCVDrawContext":
        """Creates a deep copy of the current drawing context."""
        out_obj = self.__new__(self.__class__)
        out_obj.config = self.config
        out_obj._image = self._image.copy()
        return out_obj

    @override
    def get_image(self) -> NDArray[np.uint8]:
        """Returns the current image buffer."""
        return self._image

    @override
    def draw_inplace(
        self,
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        """Draws a single triangle onto the image buffer.

        Args:
            vertices: The vertices of the triangle, an array of shape (3, 2)
                with dtype np.int32.
            color: The RGBA color of the triangle, an array of shape (4,)
                with dtype np.uint8.
        """
        check_triangle_vertices(vertices)
        check_color_rgba(color)

        alpha = color[3] / 255.0

        # If the color is fully transparent, there's nothing to draw.
        if alpha == 0.0:
            return

        # OpenCV's fillPoly expects a list of polygons.
        points = [vertices.reshape((3, 1, 2))]
        # It also expects the color as a tuple.
        rgb_color = tuple(int(c) for c in color[:3])

        # For full opacity, we can draw directly without blending.
        if alpha == 1.0:
            cv2.fillPoly(self._image, points, rgb_color)
        # For partial transparency, we need to blend with the existing image.
        else:
            overlay = self._image.copy()
            cv2.fillPoly(overlay, points, rgb_color)
            cv2.addWeighted(overlay, alpha, self._image, 1 - alpha, 0, dst=self._image)


class OpenCVDrawLossContext(DrawLossContext):
    """A drawloss context that uses the OpenCVDrawContext for rasterization."""

    @override
    def __init__(
        self,
        *,
        background_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        config: OpenCVConfig,
        **kwargs: Any,
    ) -> None:
        self.draw_context = OpenCVDrawContext(
            background_image=background_image, config=config, **kwargs
        )
        check_image_rgb(target_image)
        if background_image.shape != target_image.shape:
            raise ValueError(
                "Background and target images must have the same shape, "
                f"got {background_image.shape} and {target_image.shape}"
            )
        self._target_image = target_image
        self.config = config

    @override
    def clone(self) -> "OpenCVDrawLossContext":
        """Creates a deep copy of the current drawloss context."""
        out_obj = self.__new__(self.__class__)
        out_obj.draw_context = self.draw_context.clone()
        out_obj._target_image = self._target_image.copy()
        out_obj.config = self.config
        return out_obj

    @override
    def drawloss(
        self, vertices: NDArray[np.int32], colors: NDArray[np.uint8]
    ) -> NDArray[np.int64]:
        """Calculates the loss for drawing a triangle."""
        return calculate_drawloss_using_draw_context(
            context=self.draw_context,
            target_image=self._target_image,
            vertices=vertices,
            colors=colors,
        )


class OpenCVBackend(Backend):
    """Backend implementation using OpenCV for rasterization."""

    name: ClassVar[str] = "opencv"

    @classmethod
    @override
    def create_draw_context(
        cls: type["OpenCVBackend"],
        *,
        background_image: NDArray[np.uint8],
        config: Optional[Config] = None,
    ) -> DrawContext:
        if config is None:
            config = OpenCVConfig()
        if not isinstance(config, OpenCVConfig):
            raise TypeError(
                f"Config must be an instance of OpenCVConfig, got {type(config)}"
            )
        return OpenCVDrawContext(background_image=background_image, config=config)

    @classmethod
    @override
    def create_drawloss_context(
        cls: type["OpenCVBackend"],
        *,
        background_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        config: Optional[Config] = None,
    ) -> DrawLossContext:
        if config is None:
            config = OpenCVConfig()
        if not isinstance(config, OpenCVConfig):
            raise TypeError(
                f"Config must be an instance of OpenCVConfig, got {type(config)}"
            )
        return OpenCVDrawLossContext(
            background_image=background_image,
            target_image=target_image,
            config=config,
        )
