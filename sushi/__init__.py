"""Sushi: Optimized implementations of triangle rasterization for MSE loss."""

from sushi.backend_numpy import NumpyRasterBackend
from sushi.backend_opencv import OpenCVRasterBackend
from sushi.backend_opengl import OpenGLRasterBackend
from sushi.backend_pillow import PillowRasterBackend

try:
    from sushi.backend_cpp import CPPRasterBackend

    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

from sushi.utils import RasterBackend, np_image_mse

__all__ = [
    "RasterBackend",
    "NumpyRasterBackend",
    "OpenCVRasterBackend",
    "OpenGLRasterBackend",
    "PillowRasterBackend",
    "np_image_mse",
]

if _HAS_CPP:
    __all__.append("CPPRasterBackend")

__version__ = "0.1.0"
