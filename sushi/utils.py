from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Union

import numpy as np
from numpy.typing import NDArray


def check_image_shape(shape: Union[tuple[int, int], tuple[int, int, int]]) -> None:
    assert isinstance(shape, tuple)
    assert len(shape) in (2, 3)
    if len(shape) >= 2:
        assert shape[0] > 0
        assert shape[1] > 0
    if len(shape) == 3:
        assert shape[2] == 3  # 3-channel RGB


def check_image_rgb(image: NDArray[np.uint8]) -> None:
    assert image.ndim == 3
    assert image.shape[2] == 3  # 3-channel RGB
    assert image.dtype == np.uint8  # uint8 RGB values


def check_color_rgb(color: NDArray[np.uint8]) -> None:
    assert color.shape == (3,)
    assert color.dtype == np.uint8  # uint8 RGB values


def check_color_rgba(color: NDArray[np.uint8]) -> None:
    assert color.shape == (4,)
    assert color.dtype == np.uint8  # uint8 RGBA values


def check_triangle_vertices(vertices: NDArray[np.int32]) -> None:
    assert vertices.shape == (3, 2)
    assert vertices.dtype == np.int32  # int32 pixel coordinates


def np_image_mse(
    image1: NDArray[np.uint8],
    image2: NDArray[np.uint8],
) -> float:
    """Compute the mean squared error between two images.

    Args:
        image1: The first image, an array of shape (H, W, 3) with dtype np.uint8.
        image2: The second image, an array of shape (H, W, 3) with dtype np.uint8.

    Returns:
        The mean squared error between the two images.
    """
    check_image_rgb(image1)
    check_image_rgb(image2)
    assert image1.shape == image2.shape

    mse = np.mean((image1.astype(np.int64) - image2.astype(np.int64)) ** 2)
    return float(mse)


class RasterBackend(ABC):
    """Abstract base class for rasterization backends."""

    name: ClassVar[str]

    def __init_subclass__(cls: type["RasterBackend"], **kwargs: dict[str, Any]) -> None:
        super().__init_subclass__(**kwargs)
        # This check runs when the subclass is defined, not when it's instantiated.
        if not hasattr(cls, "name"):
            raise NotImplementedError(
                f"Class {cls.__name__} must define a 'name' class attribute."
            )

    @classmethod
    def triangle_count_pixels_single(
        cls: type["RasterBackend"],
        image_shape: tuple[int, int],
        vertices: NDArray[np.int32],
    ) -> int:
        """Count the number of pixels that would be drawn over by a triangle on an image
        of a given shape.

        Args:
            image_shape: The shape of the image, a tuple (H, W).
            vertices: The vertices of the triangle, an array of shape (3, 2)
                with dtype np.int32 representing the (x, y) coordinates of the
                triangle's corners, with the origin at the top-left corner of the image.

        Returns:
            The number of pixels that would be colored when the triangle is drawn.
        """
        sample_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        sample_color = np.array((255, 255, 255), dtype=np.uint8)
        cls.triangle_draw_single_rgb_inplace(sample_image, vertices, sample_color)
        num_colored_pixels = int(np.sum(np.any(sample_image != 0, axis=-1)))
        return num_colored_pixels

    @classmethod
    def triangle_draw_single_rgb_inplace(
        cls: type["RasterBackend"],
        image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        """Draw a triangle over a given image. The input image is modified in place.

        Args:
            image: The base RGB image, an array of shape (H, W, 3) with dtype np.uint8.
            vertices: The vertices of the triangle, an array of shape (3, 2)
                with dtype np.int32 representing the (x, y) coordinates of the
                triangle's corners, with the origin at the top-left corner of the image.
            color: The color of the triangle, an array of shape (3,) with
                dtype np.uint8, representing the RGB color of the triangle.
        """
        color = np.concatenate((color, np.array([255], dtype=np.uint8)))
        cls.triangle_draw_single_rgba_inplace(image, vertices, color)

    @classmethod
    def triangle_draw_single_rgb(
        cls: type["RasterBackend"],
        image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Draw a triangle over a given image. The input image is unmodified.

        Args:
            image: The base RGB image, an array of shape (H, W, 3) with dtype np.uint8.
            vertices: The vertices of the triangle, an array of shape (3, 2)
                with dtype np.int32 representing the (x, y) coordinates of the
                triangle's corners, with the origin at the top-left corner of the image.
            color: The color of the triangle, an array of shape (3,) with
                dtype np.uint8, representing the RGB color of the triangle.

        Returns:
            A new image array with the triangle drawn on it.
        """
        check_image_rgb(image)
        check_triangle_vertices(vertices)
        check_color_rgb(color)

        new_image = image.copy()
        cls.triangle_draw_single_rgb_inplace(new_image, vertices, color)
        return new_image

    @classmethod
    @abstractmethod
    def triangle_draw_single_rgba_inplace(
        cls: type["RasterBackend"],
        image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        """Draw a triangle with alpha blending over a given image.
        The input image is modified in place.

        Args:
            image: The base RGB image, an array of shape (H, W, 3) with dtype np.uint8.
            vertices: The vertices of the triangle, an array of shape (3, 2)
                with dtype np.int32 representing the (x, y) coordinates of the
                triangle's corners, with the origin at the top-left corner of the image.
            color: The color of the triangle, an array of shape (4,) with
                dtype np.uint8, representing the RGBA color of the triangle.
        """
        pass

    @classmethod
    def triangle_draw_single_rgba(
        cls: type["RasterBackend"],
        image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Draw a triangle with alpha blending over a given image.
        The input image is unmodified.

        Args:
            image: The base RGB image, an array of shape (H, W, 3) with dtype np.uint8.
            vertices: The vertices of the triangle, an array of shape (3, 2)
                with dtype np.int32 representing the (x, y) coordinates of the
                triangle's corners, with the origin at the top-left corner of the image.
            color: The color of the triangle, an array of shape (4,) with
                dtype np.uint8, representing the RGBA color of the triangle.

        Returns:
            A new image array with the triangle drawn on it.
        """
        check_image_rgb(image)
        check_triangle_vertices(vertices)
        check_color_rgba(color)

        new_image = image.copy()
        cls.triangle_draw_single_rgba_inplace(new_image, vertices, color)
        return new_image

    @staticmethod
    def _compute_drawloss(
        image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        modified_image: NDArray[np.uint8],
        base_loss: Optional[float] = None,
    ) -> float:
        """Compute the MSE loss delta between a collection of images, after 'drawing'
        onto the base image.

        Args:
            image: The base RGB image, an array of shape (H, W, 3) with dtype np.uint8.
            target_image: The target RGB image, of same type and shape as `image`.
            modified_image: The modified RGB image after drawing,
                of same type and shape as `image`.
            base_loss: If provided, the MSE loss between the base and the target image.
                If not provided, it will be computed.

        Returns:
            The MSE loss delta `x` such that:
            `MSE(modified_image, target_image) == MSE(image, target_image) + x`.
        """
        check_image_rgb(image)
        check_image_rgb(target_image)
        check_image_rgb(modified_image)
        assert image.shape == target_image.shape
        assert image.shape == modified_image.shape

        # Compute the MSE loss between the modified image and the target image.
        if base_loss is None:
            base_loss = float(
                np.mean((image.astype(np.int64) - target_image.astype(np.int64)) ** 2)
            )

        assert base_loss is not None
        assert base_loss >= 0.0, "Base MSE should be non-negative."

        modified_loss = np.mean(
            (modified_image.astype(np.int64) - target_image.astype(np.int64)) ** 2
        )

        return float(modified_loss - base_loss)

    @classmethod
    def triangle_drawloss_single_rgb(
        cls: type["RasterBackend"],
        image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
        base_loss: Optional[float] = None,
    ) -> float:
        """Calculate the MSE loss delta that would be incurred by drawing a triangle
        over a given image, compared to a target image. The input image is unmodified.

        Args:
            image: The base RGB image, an array of shape (H, W, 3) with dtype np.uint8.
            target_image: The target RGB image, of same type and shape as `image`.
            vertices: The vertices of the triangle, an array of shape (3, 2)
                with dtype np.int32 representing the (x, y) coordinates of the
                triangle's corners, with the origin at the top-left corner of the image.
            color: The color of the triangle, an array of shape (3,) with
                dtype np.uint8, representing the RGB color of the triangle.
            base_loss: If provided, the MSE loss between the base and the target image.
                If not provided, it will be computed.

        Returns:
            The MSE loss delta `x` such that:
            `MSE(draw(triangle, image), target_image) == MSE(image, target_image) + x`.
        """
        modified_image = cls.triangle_draw_single_rgb(image, vertices, color)
        return cls._compute_drawloss(image, target_image, modified_image, base_loss)

    @classmethod
    def triangle_drawloss_single_rgba(
        cls: type["RasterBackend"],
        image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
        base_loss: Optional[float] = None,
    ) -> float:
        """Calculate the MSE loss delta that would be incurred by drawing a triangle
        over a given image, compared to a target image. The input image is unmodified.

        Args:
            image: The base RGB image, an array of shape (H, W, 3) with dtype np.uint8.
            target_image: The target RGB image, of same type and shape as `image`.
            vertices: The vertices of the triangle, an array of shape (3, 2)
                with dtype np.int32 representing the (x, y) coordinates of the
                triangle's corners, with the origin at the top-left corner of the image.
            color: The color of the triangle, an array of shape (4,) with dtype np.uint8
                representing the RGBA color of the triangle.
            base_loss: If provided, the MSE loss between the base and the target image.
                If not provided, it will be computed.

        Returns:
            The MSE loss delta `x` such that:
            `MSE(draw(triangle, image), target_image) == MSE(image, target_image) + x`.
        """
        modified_image = cls.triangle_draw_single_rgba(image, vertices, color)
        return cls._compute_drawloss(image, target_image, modified_image, base_loss)

    @classmethod
    def triangle_drawloss_batch_rgba(
        cls: type["RasterBackend"],
        image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        colors: NDArray[np.uint8],
        base_loss: Optional[float] = None,
    ) -> NDArray[np.float32]:
        """Calculate the MSE loss delta that would be incurred by drawing a batch of
        triangles over a given image, compared to a target image. The input image is
        unmodified. The scores are computed independently for each triangle, equivalent
        to a loop over `triangle_drawloss_single_rgba`.

        Args:
            image: The base RGB image, an array of shape (H, W, 3) with dtype np.uint8.
            target_image: The target RGB image, of same type and shape as `image`.
            vertices: The vertices of the triangles, an array of shape (N, 3, 2)
                with dtype np.int32 representing the (x, y) coordinates of the
                triangles' corners, with the origin at the top-left corner of the image.
            colors: The colors of the triangles, an array of shape (N, 4) with dtype np.uint8
                representing the RGBA colors of the triangles.
            base_loss: If provided, the MSE loss between the base and the target image.
                If not provided, it will be computed.

        Returns:
            An array of shape (N,) with the MSE loss deltas for each triangle.
        """
        return np.array(
            [
                cls.triangle_drawloss_single_rgba(
                    image, target_image, vertices[i], colors[i], base_loss
                )
                for i in range(vertices.shape[0])
            ]
        )
