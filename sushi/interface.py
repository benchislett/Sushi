from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    Self,
)

import numpy as np
from numpy.typing import NDArray

from sushi.utils import (
    np_image_loss,
)


class Config(ABC):
    """Abstract base class for context configuration objects."""

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

    @abstractmethod
    def __init__(
        self,
        *,
        background_image: NDArray[np.uint8],
        config: Config,
        **kwargs: Any,
    ) -> None:
        """Initialize the drawing context.

        Args:
            background_image: The background image as a NumPy array of shape (H, W, 3)
                with dtype np.uint8.
            config: A configuration object specific to the backend.

        Raises:
            ValueError: If the image does not have the correct shape or type, or if
                the config does not support drawing operations.
        """
        pass

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

    @abstractmethod
    def __init__(
        self,
        *,
        background_image: Optional[NDArray[np.uint8]],
        target_image: Optional[NDArray[np.uint8]],
        config: Config,
        **kwargs: Any,
    ) -> None:
        """Initialize the draw loss context.

        Args:
            background_image: The background image as a NumPy array of shape (H, W, 3)
                with dtype np.uint8.
            target_image: The target image as a NumPy array of shape (H, W, 3) with
                dtype np.uint8.
            config: A configuration object specific to the backend.

        Raises:
            ValueError: If the images do not have the correct shape or type, or if
                the config does not support drawloss operations.
        """
        pass

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
        computing the output values independently for each.
        For scalar operation, append a singleton dimension to the inputs.

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


def calculate_drawloss_using_draw_context(
    context: DrawContext,
    target_image: NDArray[np.uint8],
    vertices: NDArray[np.int32],
    colors: NDArray[np.uint8],
) -> NDArray[np.int64]:
    """Compute the change in Sum of Squares Error loss between the context's
    background image and target image that would result from drawing a triangle
    with the given vertices and color atop the background image.

    This implementation calls `draw()` and computes the loss using NumPy operations.

    Args:
        context: A DrawContext instance.
        target_image: The target image as a NumPy array of shape (H, W, 3) with
            dtype np.uint8.
        vertices: The vertices of the triangles, an array of shape (N, 3, 2)
            with dtype np.int32 representing the (x, y) coordinates of the
            triangles' corners, with the origin at the top-left corner of the image.
        color: The colors of the triangles, an array of shape (N, 4) with
            dtype np.uint8 representing the RGBA colors.

    Returns:
        An array of shape (N,) with dtype np.int64 representing the change in
        Sum of Squares Error loss for each triangle.
    """
    num_triangles = vertices.shape[0]
    losses = np.empty((num_triangles,), dtype=np.int64)

    base_loss = np_image_loss(context.get_image(), target_image)

    for i in range(num_triangles):
        new_image = context.draw(vertices[i], colors[i])
        losses[i] = np_image_loss(new_image, target_image) - base_loss

    return losses


class Backend(ABC):
    """Abstract base class for backends."""

    name: ClassVar[str] = "abc"
    """Name of the backend."""

    @classmethod
    @abstractmethod
    def create_draw_context(
        cls: type["Backend"],
        *,
        background_image: NDArray[np.uint8],
        config: Optional["Config"] = None,
    ) -> "DrawContext":
        """Create a drawing context for this backend. This method will return
        different context implementations depending on the provided configuration.

        Args:
            background_image: The background image as a NumPy array of shape (H, W, 3)
                with dtype np.uint8.
            config: A configuration object specific to the backend, or None to use
                a default configuration.

        Returns:
            A drawing context for this backend.
        """
        pass

    @classmethod
    @abstractmethod
    def create_drawloss_context(
        cls: type["Backend"],
        *,
        background_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        config: Optional["Config"] = None,
    ) -> "DrawLossContext":
        """Create a draw loss context for this backend. This method will return
        different context implementations depending on the provided configuration.

        The configuration must be one of the implementations returned by
        `list_implementations()` and must have `supports_drawloss` set to True.

        Args:
            background_image: The background image as a NumPy array of shape (H, W, 3)
                with dtype np.uint8.
            target_image: The target image as a NumPy array of shape (H, W, 3) with
                dtype np.uint8.
            config: A configuration object specific to the backend, or None to use
                a default configuration.

        Returns:
            A draw loss context for this backend.
        """
        pass

    @classmethod
    def is_available(cls: type["Backend"]) -> bool:
        """Check if the backend is available on the current system."""
        return True

    @classmethod
    def is_mode_supported(
        cls: type["Backend"], mode: Literal["draw", "drawloss"]
    ) -> tuple[int, Optional[str]]:
        """Check if the backend supports the given mode.

        Args:
            mode: The mode to check, either "draw" or "drawloss".

        Returns:
            A tuple (support_level, error_message) where:

            - support_level is:
                - 1 if the mode is supported,
                - 0 if the mode is not supported,
                - -1 if an error occurred while checking. This can happen if the
                    backend is partially installed, misconfigured, or unavailable.
            - error_message is None if support_level is 1 or 0, or a string
              describing the error if support_level is -1.
        """
        if mode == "draw":
            try:
                _ = cls.create_draw_context(
                    background_image=np.zeros((10, 10, 3), dtype=np.uint8)
                )
                return 1, None
            except NotImplementedError:
                return 0, None
            except Exception as e:
                return -1, str(e)
        elif mode == "drawloss":
            try:
                _ = cls.create_drawloss_context(
                    background_image=np.zeros((10, 10, 3), dtype=np.uint8),
                    target_image=np.zeros((10, 10, 3), dtype=np.uint8),
                )
                return 1, None
            except NotImplementedError:
                return 0, None
            except Exception as e:
                return -1, str(e)
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Supported modes are 'draw' and 'drawloss'."
            )
