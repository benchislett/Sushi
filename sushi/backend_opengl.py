from typing import ClassVar

import moderngl
import numpy as np
from numpy.typing import NDArray

from sushi.utils import (
    RasterBackend,
    check_color_rgba,
    check_image_rgb,
    check_triangle_vertices,
)


class OpenGLRasterBackend(RasterBackend):
    name: ClassVar[str] = "opengl"

    @classmethod
    def triangle_draw_single_rgba_inplace(
        cls: type["OpenGLRasterBackend"],
        image: NDArray[np.uint8],
        vertices: NDArray[np.int32],
        color: NDArray[np.uint8],
    ) -> None:
        check_image_rgb(image)
        check_triangle_vertices(vertices)
        check_color_rgba(color)

        H, W, _ = image.shape
        ctx = moderngl.create_standalone_context()

        prog = ctx.program(
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

        vertices_gl = vertices.astype(np.float32) + 0.5
        vertices_gl[:, 0] = (vertices_gl[:, 0] / W) * 2 - 1
        vertices_gl[:, 1] = (vertices_gl[:, 1] / H) * -2 + 1

        color_f32 = (color / 255.0).astype(np.float32)
        prog["in_color"] = tuple(color_f32)

        vbo = ctx.buffer(vertices_gl.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, "2f", "in_vert")])

        image_f32 = (image / 255.0).astype(np.float32)

        texture = ctx.texture((W, H), 3, data=image_f32.tobytes(), dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[texture])
        fbo.use()

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        vao.render(moderngl.TRIANGLES)

        rendered_bytes = fbo.read(components=3, dtype="f4")
        rendered_image_f32 = np.frombuffer(rendered_bytes, dtype=np.float32).reshape(
            (H, W, 3)
        )

        final_image_uint8 = (np.clip(rendered_image_f32, 0.0, 1.0) * 255).astype(
            np.uint8
        )

        np.copyto(image, np.flipud(final_image_uint8))

        # Clean up OpenGL resources
        fbo.release()
        texture.release()
        vao.release()
        vbo.release()
        prog.release()
        ctx.release()
