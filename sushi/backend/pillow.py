from typing import Any, ClassVar, Optional, override

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

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


class PillowConfig(Config):
    pass


class PillowDrawContext(DrawContext):
    """A drawing context that uses the ImageDraw module from the Pillow library."""

    def __init__(
        self,
        *,
        background_image: NDArray[np.uint8],
        config: PillowConfig,
    ) -> None:
        check_image_rgb(background_image)
        self.config = config

        self._pil_background = Image.fromarray(background_image).convert("RGB")
        self.pil_draw_context = ImageDraw.Draw(self._pil_background, "RGBA")

    @override
    def clone(self) -> "PillowDrawContext":
        out_obj = self.__new__(self.__class__)
        out_obj.config = self.config
        out_obj._pil_background = self._pil_background.copy()
        out_obj.pil_draw_context = ImageDraw.Draw(out_obj._pil_background, "RGBA")
        return out_obj

    @override
    def get_image(self) -> NDArray[np.uint8]:
        return np.array(self._pil_background, dtype=np.uint8)

    @override
    def draw_inplace(
        self,
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        check_triangle_vertices(vertices)
        check_color_rgba(color)
        self.pil_draw_context.polygon([tuple(v) for v in vertices], fill=tuple(color))


class PillowDrawLossContext(DrawLossContext):
    """A drawloss context that uses the ImageDraw module from the Pillow library"""

    @override
    def __init__(
        self,
        *,
        background_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        config: PillowConfig,
        **kwargs: Any,
    ) -> None:
        self.draw_context = PillowDrawContext(
            background_image=background_image, config=config, **kwargs
        )
        check_image_rgb(target_image)
        if background_image.shape != target_image.shape:
            raise ValueError(
                "Background image and target image must have the same shape, "
                f"got {background_image.shape} and {target_image.shape}"
            )
        self._target_image_np = target_image

    @override
    def clone(self) -> "PillowDrawLossContext":
        out_obj = self.__new__(self.__class__)
        out_obj.draw_context = self.draw_context.clone()
        out_obj._target_image_np = self._target_image_np.copy()
        return out_obj

    @override
    def drawloss(
        self, vertices: NDArray[np.int32], colors: NDArray[np.uint8]
    ) -> NDArray[np.int64]:
        return calculate_drawloss_using_draw_context(
            context=self.draw_context,
            target_image=self._target_image_np,
            vertices=vertices,
            colors=colors,
        )


class PillowBackend(Backend):
    name: ClassVar[str] = "pillow"

    @classmethod
    @override
    def create_draw_context(
        cls: type["PillowBackend"],
        *,
        background_image: NDArray[np.uint8],
        config: Optional[Config] = None,
    ) -> DrawContext:
        if config is None:
            config = PillowConfig()
        if not isinstance(config, PillowConfig):
            raise TypeError(
                f"Config must be an instance of PillowConfig, got {type(config)}"
            )
        return PillowDrawContext(background_image=background_image, config=config)

    @classmethod
    @override
    def create_drawloss_context(
        cls: type["PillowBackend"],
        *,
        background_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        config: Optional[Config] = None,
    ) -> DrawLossContext:
        if config is None:
            config = PillowConfig()
        if not isinstance(config, PillowConfig):
            raise TypeError(
                f"Config must be an instance of PillowConfig, got {type(config)}"
            )
        return PillowDrawLossContext(
            background_image=background_image,
            target_image=target_image,
            config=config,
        )
