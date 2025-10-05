from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Union, override

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from sushi.interface import (
    Backend,
    Config,
    Context,
    requires_count_pixels_support,
    requires_draw_support,
    requires_drawloss_support,
    requires_target_image,
)
from sushi.utils import (
    check_color_rgb,
    check_color_rgba,
    check_image_rgb,
    check_image_shape,
    check_triangle_vertices,
    np_image_loss,
)


class PillowBackend(ABC):
    name: ClassVar[str] = "pillow"

    @override
    @classmethod
    def list_implementations(cls: type["Backend"]) -> list[type["Config"]]:
        """List all available configuration implementations for this backend.

        Returns:
            A list of configuration classes that can be used with this backend.
        """
        return [PillowConfig]

    @override
    @classmethod
    def create_context(
        cls: type["Backend"],
        background_image: NDArray[np.uint8],
        target_image: Optional[NDArray[np.uint8]] = None,
        config: Optional[Config] = None,
    ) -> Context:
        if config is None:
            config = PillowConfig()

        if not isinstance(config, PillowConfig):
            raise TypeError(
                f"Config must be an instance of PillowConfig, got {type(config)}"
            )
        return PillowContext(background_image, target_image=target_image, config=config)


class PillowConfig(Config):
    pass


class PillowContext(Context):
    name: ClassVar[str] = "context_pillow"

    supports_draw: ClassVar[bool] = True
    supports_drawloss: ClassVar[bool] = True
    supports_count_pixels: ClassVar[bool] = True

    def __init__(
        self,
        background_image: NDArray[np.uint8],
        *,
        target_image: Optional[NDArray[np.uint8]] = None,
        config: PillowConfig,
    ) -> None:
        super().__init__(
            background_image=background_image,
            target_image=target_image,
            config=config,
        )
        self.pil_background = Image.fromarray(self.background_image, mode="RGB")
        if self.target_image is not None:
            self.pil_target = Image.fromarray(self.target_image, mode="RGB")

        if not isinstance(config, PillowConfig):
            raise TypeError(
                f"Config must be an instance of PillowConfig, got {type(config)}"
            )

    @override
    @requires_draw_support
    def draw_inplace(
        self,
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        check_color_rgba(color)
        draw = ImageDraw.Draw(self.pil_background, "RGBA")
        draw.polygon([tuple(v) for v in vertices], fill=tuple(color))
        self.background_image = np.array(self.pil_background, dtype=np.uint8)

    @override
    @requires_draw_support
    def draw(
        self,
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        check_color_rgba(color)
        new_image = self.pil_background.copy()
        draw = ImageDraw.Draw(new_image, "RGBA")
        draw.polygon([tuple(v) for v in vertices], fill=tuple(color))
        return np.array(new_image, dtype=np.uint8)

    @override
    @requires_count_pixels_support
    def count_pixels(
        self,
        vertices: NDArray[np.int32],
    ) -> int:
        blank_image = Image.new("L", self.pil_background.size, 0)
        draw = ImageDraw.Draw(blank_image, "L")
        draw.polygon([tuple(v) for v in vertices], fill=1)
        mask = np.array(blank_image, dtype=np.uint8)
        return int(np.sum(mask))

    @override
    @requires_drawloss_support
    @requires_target_image
    def drawloss(
        self,
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> NDArray[np.int64]:
        current_loss = np_image_loss(self.background_image, self.target_image)

        losses = np.empty(len(vertices), dtype=np.int64)
        for i, (v, c) in enumerate(zip(vertices, color)):
            check_triangle_vertices(v)
            check_color_rgba(c)
            out = self.draw(v, c)
            new_loss = np_image_loss(out, self.target_image)
            losses[i] = new_loss - current_loss

        return losses
