import functools
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, cast, override

import moderngl
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


class OpenGLConfig(Config):
    """Configuration for the OpenGL backend."""

    pass


class OpenGLDrawContext(DrawContext):
    """A drawing context that uses ModernGL for hardware-accelerated rasterization."""

    def __init__(
        self,
        *,
        background_image: NDArray[np.uint8],
        config: OpenGLConfig,
    ) -> None:
        """Initializes the OpenGL context and resources for drawing.

        Args:
            background_image: The initial image to draw on.
            config: The configuration for the OpenGL context.
        """
        check_image_rgb(background_image)
        self.config = config
        self._released = False

        H, W, _ = background_image.shape
        self._shape = (H, W)

        # Initialize OpenGL context and resources
        self.ctx = moderngl.create_standalone_context()

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec4 in_color;
                out vec4 f_color;
                void main() {
                    f_color = in_color;
                }
            """,
        )

        # We create a single VBO that will be updated for each triangle
        self.vbo = self.ctx.buffer(reserve=3 * 2 * 4)  # 3 vertices, 2 floats, 4 bytes
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_vert")])

        # Create a texture and framebuffer to draw onto
        image_f32 = (background_image / 255.0).astype("f4")
        self.texture = self.ctx.texture((W, H), 3, data=image_f32.tobytes(), dtype="f4")
        self.fbo = self.ctx.framebuffer(color_attachments=[self.texture])
        self.fbo.use()

        # Enable alpha blending
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self) -> None:
        """Releases all associated OpenGL resources."""
        if not self._released and hasattr(self, "ctx"):
            self.fbo.release()
            self.texture.release()
            self.vao.release()
            self.vbo.release()
            self.prog.release()
            self.ctx.release()
            self._released = True

    def __del__(self) -> None:
        self.release()

    @override
    def clone(self) -> "OpenGLDrawContext":
        """Creates a new OpenGLDrawContext with a copy of the current image.

        Note: This is not a cheap operation, as it involves reading the image
        from the GPU back to the CPU and then re-uploading it to a new context.
        """
        current_image = self.get_image()
        return OpenGLDrawContext(background_image=current_image, config=self.config)

    @override
    def get_image(self) -> NDArray[np.uint8]:
        """Reads the current image from the GPU framebuffer and returns it.

        Returns:
            The current state of the image as an (H, W, 3) NumPy array.
        """
        self.fbo.use()
        H, W = self._shape
        rendered_bytes = self.fbo.read(components=3, dtype="f4")
        image_f32 = np.frombuffer(rendered_bytes, dtype=np.float32).reshape((H, W, 3))

        # Convert back to uint8 and flip vertically to match NumPy's coordinate system
        image_uint8 = (np.clip(image_f32, 0.0, 1.0) * 255).astype(np.uint8)
        return np.flipud(image_uint8)

    @override
    def draw_inplace(
        self,
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        """Draws a single colored triangle onto the image.

        Args:
            vertices: A (3, 2) array of triangle vertex coordinates.
            color: A (4,) array representing the RGBA color.
        """
        check_triangle_vertices(vertices)
        check_color_rgba(color)

        H, W = self._shape

        # Convert vertices to OpenGL's normalized device coordinates [-1, 1]
        vertices_gl = vertices.astype(np.float32) + 0.5
        vertices_gl[:, 0] = (vertices_gl[:, 0] / W) * 2.0 - 1.0
        vertices_gl[:, 1] = (vertices_gl[:, 1] / H) * -2.0 + 1.0  # Flip Y-axis

        # Convert color to float [0, 1] and set as a uniform
        color_f32 = (color / 255.0).astype(np.float32)
        self.prog["in_color"] = tuple(color_f32)

        # Update VBO, bind FBO, and render
        self.vbo.write(vertices_gl.tobytes())
        self.fbo.use()
        self.vao.render(moderngl.TRIANGLES)


class OpenGLDrawLossContext(DrawLossContext):
    """A drawloss context that uses the OpenGLDrawContext for rasterization."""

    @override
    def __init__(
        self,
        *,
        background_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        config: OpenGLConfig,
        **kwargs: Any,
    ) -> None:
        self.draw_context = OpenGLDrawContext(
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
    def clone(self) -> "OpenGLDrawLossContext":
        out_obj = self.__new__(self.__class__)
        out_obj.draw_context = self.draw_context.clone()
        out_obj._target_image = self._target_image.copy()
        out_obj.config = self.config
        return out_obj

    @override
    def drawloss(
        self, vertices: NDArray[np.int32], colors: NDArray[np.uint8]
    ) -> NDArray[np.int64]:
        return calculate_drawloss_using_draw_context(
            context=self.draw_context,
            target_image=self._target_image,
            vertices=vertices,
            colors=colors,
        )


@functools.lru_cache(maxsize=1)
def is_opengl_available() -> bool:
    try:
        ctx = moderngl.create_standalone_context()
        ctx.release()
        return True
    except Exception:
        return False


class OpenGLBackend(Backend):
    name: ClassVar[str] = "opengl"

    @classmethod
    @override
    def create_draw_context(
        cls: type["OpenGLBackend"],
        *,
        background_image: NDArray[np.uint8],
        config: Optional[Config] = None,
    ) -> DrawContext:
        if config is None:
            config = OpenGLConfig()
        if not isinstance(config, OpenGLConfig):
            raise TypeError(
                f"Config must be an instance of OpenGLConfig, got {type(config)}"
            )
        return OpenGLDrawContext(background_image=background_image, config=config)

    @classmethod
    @override
    def create_drawloss_context(
        cls: type["OpenGLBackend"],
        *,
        background_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        config: Optional[Config] = None,
    ) -> DrawLossContext:
        if config is None:
            config = OpenGLConfig()
        if not isinstance(config, OpenGLConfig):
            raise TypeError(
                f"Config must be an instance of OpenGLConfig, got {type(config)}"
            )
        return OpenGLDrawLossContext(
            background_image=background_image,
            target_image=target_image,
            config=config,
        )

    @classmethod
    @override
    def is_available(cls: type["OpenGLBackend"]) -> bool:
        return is_opengl_available()
