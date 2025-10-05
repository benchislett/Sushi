from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Optional, Self, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from sushi.utils import (
    check_color_rgb,
    check_color_rgba,
    check_image_rgb,
    check_image_shape,
    check_triangle_vertices,
    np_image_loss,
)


class Config(ABC):
    """Abstract base class for context configuration objects."""

    supports_draw: ClassVar[bool] = False
    """Whether the configuration is compatible with DrawContext usage."""

    supports_drawloss: ClassVar[bool] = False
    """Whether the configuration is compatible with DrawLossContext usage."""

    @abstractmethod
    def name(self) -> str:
        """Get the name of the configuration.

        Returns:
            The name of the configuration.
        """
        pass


class Context(ABC):
    """Abstract base class for rendering context."""

    @abstractmethod
    def clone(self) -> "Self":
        """
        Creates a new context with an identical but independent copy of the internal
        state.

        Backends must implement this to perform safe and efficient copying
        (e.g., on-device data for GPU contexts).
        """
        pass


class DrawContext(Context):
    """Abstract base class for rendering contexts that support drawing triangles.

    The state of the context can be modified by drawing triangles onto it using
    `draw_inplace()`. To access the current image, use `get_image()`.

    Alternatively, triangles can be drawn without modifying the context's state
    using `draw()`, which returns a new image with the triangle drawn on it.
    Performing `draw()` may involve copying the context's internal state,
    so it is generally less efficient than `draw_inplace()`.
    """

    def __init__(
        self,
        *,
        background_image: NDArray[np.uint8],
        config: Config,
    ) -> None:
        """Initialize the drawing context.

        Args:
            background_image: The background image as a NumPy array of shape (H, W, 3)
                with dtype np.uint8.
            config: A configuration object specific to the backend.

        Raises:
            ValueError: If the images do not have the correct shape or type.
        """
        check_image_rgb(background_image)
        self.config = config

    @abstractmethod
    def get_image(self) -> NDArray[np.uint8]:
        """Get the current background image.

        Returns:
            The current background image as a NumPy array of shape (H, W, 3)
            with dtype np.uint8.
        """
        pass

    @abstractmethod
    def draw_inplace(
        self,
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        """Draw a triangle onto the context's background image in-place.
        The context's state will be modified.

        Args:
            vertices: The vertices of the triangle, an array of shape (3, 2)
                with dtype np.int32 representing the (x, y) coordinates of the
                triangle's corners, with the origin at the top-left corner of the image.
            color: The color of the triangle, an array of shape (4,) with
                dtype np.uint8 representing the RGBA color.
        """
        pass

    def draw(
        self,
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Draw a triangle onto the context's background image and return a new image.
        The context's state will not be modified.

        By default, this method creates a clone of the context and performs
        `draw_inplace()` on the clone. Backends may override this method to
        provide a more efficient implementation.

        Args:
            vertices: The vertices of the triangle, an array of shape (3, 2)
                with dtype np.int32 representing the (x, y) coordinates of the
                triangle's corners, with the origin at the top-left corner of the image.
            color: The color of the triangle, an array of shape (4,) with
                dtype np.uint8 representing the RGBA color.

        Returns:
            A new image as a NumPy array of shape (H, W, 3) with dtype np.uint8
            representing the background image with the triangle drawn on it.
        """
        temp_context = self.clone()
        temp_context.draw_inplace(vertices, color)
        return temp_context.get_image()


class DrawLossContext(Context):
    """Abstract base class for rendering contexts that support mutation-free
    Sum of Squared Errors (SSE) fused draw+loss computation.

    See `drawloss()` for details.
    """

    def __init__(
        self,
        *,
        background_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        config: Config,
    ) -> None:
        """Initialize the draw loss context.

        Args:
            background_image: The background image as a NumPy array of shape (H, W, 3)
                with dtype np.uint8.
            target_image: The target image as a NumPy array of shape (H, W, 3) with
                dtype np.uint8. Must be provided if the backend supports drawloss.
            config: A configuration object specific to the backend.

        Raises:
            ValueError: If the images do not have the correct shape or type.
        """
        check_image_rgb(background_image)
        check_image_rgb(target_image)
        if background_image.shape != target_image.shape:
            msg = "Target image must have the same shape as background image."
            raise ValueError(msg)

        self.config = config

    @abstractmethod
    def drawloss(
        self,
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> NDArray[np.int64]:
        """Compute the change in Sum of Squares Error loss between the context's
        background image and target image that would result from drawing a triangle
        with the given vertices and color atop the background image.

        This is a batch operation, taking an array of triangles and colors, and
        computing the output values independently for each. For scalar operation,
        just pass a single-element array.

        This method does not mutate the context's background or target images, but
        it may temporarily mutate them during its computation.
        It is therefore not guaranteed to be thread-safe.

        Args:
            vertices: The vertices of the triangles, an array of shape (N, 3, 2)
                with dtype np.int32 representing the (x, y) coordinates of the
                triangles' corners, with the origin at the top-left corner of the image.
            color: The colors of the triangles, an array of shape (N, 4) with
                dtype np.uint8 representing the RGBA colors.

        Returns:
            An array of shape (N,) with dtype np.int64 representing the change in
            Sum of Squares Error loss for each triangle.
        """
        pass


class Backend(ABC):
    """Abstract base class for backends."""

    name: ClassVar[str] = "abc"
    """Name of the backend."""

    @abstractmethod
    @classmethod
    def list_implementations(cls: type["Backend"]) -> list[type["Config"]]:
        """List all available configuration implementations for this backend.

        Configuration classes define `supports_draw` and `supports_drawloss`
        class variables to indicate whether they are compatible with the respective
        context types, see `Backend.create_draw_context()` and
        `Backend.create_drawloss_context()`.

        Returns:
            A list of configuration classes that can be used with this backend.
        """
        pass

    # @abstractmethod
    # @classmethod
    # def create_draw_context

    @abstractmethod
    @classmethod
    def create_context(
        cls: type["Backend"],
        background_image: NDArray[np.uint8],
        target_image: Optional[NDArray[np.uint8]] = None,
        config: Optional["Config"] = None,
    ) -> "Context":
        """Create a rendering context for this backend. This method will return
        different context implementations depending on the provided configuration.

        The configuration must be one of the implementations returned by
        `list_implementations()`.

        Args:
            background_image: The background image as a NumPy array of shape (H, W, 3)
                with dtype np.uint8.
            target_image: The target image as a NumPy array of shape (H, W, 3) with
                dtype np.uint8. Must be provided if the backend supports drawloss.
            config: A configuration object specific to the backend, or None to use
                a default configuration.

        Returns:
            A rendering context for this backend.
        """
        pass


# class Context(ABC):
#     """Abstract base class for rendering context."""

#     name: ClassVar[str] = "abc"
#     """Name of the context."""

#     supports_draw: ClassVar[bool] = False
#     """Whether the context supports drawing triangles."""

#     supports_drawloss: ClassVar[bool] = False
#     """Whether the context supports mutation-free MSE draw loss computation."""

#     supports_count_pixels: ClassVar[bool] = False
#     """Whether the context supports counting affected pixels."""

#     def __init__(
#         self,
#         background_image: NDArray[np.uint8],
#         *,
#         target_image: Optional[NDArray[np.uint8]] = None,
#         config: Config,
#     ) -> None:
#         """Initialize the rendering context.

#         Args:
#             background_image: The background image as a NumPy array of shape (H, W, 3)
#                 with dtype np.uint8.
#             target_image: The target image as a NumPy array of shape (H, W, 3) with
#                 dtype np.uint8. Must be provided
#             config: A configuration object specific to the backend.

#         Raises:
#             ValueError: If the images do not have the correct shape or type.
#         """
#         assert background_image is not None, "Background image must be provided."
#         check_image_shape(background_image)
#         check_image_rgb(background_image)
#         self.background_image = background_image
#         self.height, self.width, _ = background_image.shape

#         if target_image is not None:
#             if not self.supports_drawloss:
#                 raise ValueError(
#                     f"{self.__class__.__name__} does not support target images."
#                 )
#             check_image_shape(target_image)
#             check_image_rgb(target_image)
#             if target_image.shape != background_image.shape:
#                 msg = "Target image must have the same shape as background image."
#                 raise ValueError(msg)

#         self.target_image = target_image

#         self.config = config

#     @abstractmethod
#     @requires_draw_support
#     def draw_inplace(
#         self,
#         vertices: NDArray[np.int32],
#         color: NDArray[np.uint8],
#     ) -> None:
#         """Draw a triangle onto the context's background image in-place.
#         The background image will be modified.

#         Args:
#             vertices: The vertices of the triangle, an array of shape (3, 2)
#                 with dtype np.int32 representing the (x, y) coordinates of the
#                 triangle's corners, with the origin at the top-left corner of the image.
#             color: The color of the triangle, an array of shape (4,) with
#                 dtype np.uint8 representing the RGBA color.
#         """
#         pass

#     @abstractmethod
#     @requires_draw_support
#     def draw(
#         self,
#         vertices: NDArray[np.int32],
#         color: NDArray[np.uint8],
#     ) -> NDArray[np.uint8]:
#         """Draw a triangle onto the context's background image and return a new image.
#         The background image will not be modified.

#         Args:
#             vertices: The vertices of the triangle, an array of shape (3, 2)
#                 with dtype np.int32 representing the (x, y) coordinates of the
#                 triangle's corners, with the origin at the top-left corner of the image.
#             color: The color of the triangle, an array of shape (4,) with
#                 dtype np.uint8 representing the RGBA color.

#         Returns:
#             A new image as a NumPy array of shape (H, W, 3) with dtype np.uint8
#             representing the background image with the triangle drawn on it.
#         """
#         pass

#     @abstractmethod
#     @requires_count_pixels_support
#     def count_pixels(
#         self,
#         vertices: NDArray[np.int32],
#     ) -> int:
#         """Count the number of pixels that would be affected by drawing a triangle.

#         Generally, this method is used for debugging and is not guaranteed to be
#         well-optimized.

#         This method does not mutate the context's background or target images, but
#         it may temporarily replace them during its computation.
#         It is therefore not thread-safe.

#         Args:
#             vertices: The vertices of the triangle, an array of shape (3, 2)
#                 with dtype np.int32 representing the (x, y) coordinates of the
#                 triangle's corners, with the origin at the top-left corner of the image.

#         Returns:
#             The number of pixels that would be affected by drawing the triangle.
#         """
#         pass

#     @abstractmethod
#     @requires_drawloss_support
#     @requires_target_image
#     def drawloss(
#         self,
#         vertices: NDArray[np.int32],
#         color: NDArray[np.uint8],
#     ) -> NDArray[np.int64]:
#         """Compute the change in Sum of Squares Error loss between the context's
#         background image and target image that would result from drawing a triangle
#         with the given vertices and color atop the background image.

#         This is a batch operation, taking an array of triangles and colors, and
#         computing the output values independently for each. For scalar operation,
#         pass a single-element array.

#         This method does not mutate the context's background or target images, but
#         it may temporarily replace them during its computation.
#         It is therefore not guaranteed to be thread-safe.

#         Args:
#             vertices: The vertices of the triangles, an array of shape (N, 3, 2)
#                 with dtype np.int32 representing the (x, y) coordinates of the
#                 triangles' corners, with the origin at the top-left corner of the image.
#             color: The colors of the triangles, an array of shape (N, 4) with
#                 dtype np.uint8 representing the RGBA colors.

#         Returns:
#             An array of shape (N,) with dtype np.int64 representing the change in
#             Sum of Squares Error loss for each triangle.
#         """
#         pass
