#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <sushi/rasterizer.h>

#ifdef WITH_CUDA
#include <sushi_cuda/cuda_backend.cuh>
#endif

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(sushi_core, m) {
    m.doc() = "Sushi C++ rasterization backend";

    /*
    Stateless class providing CPU-based rasterization functionalities.
    */
    nb::class_<CPPRasterBackend>(m, "CPPRasterBackend")
        .def_static("triangle_draw_single_rgba_inplace",
                   &CPPRasterBackend::triangle_draw_single_rgba_inplace,
                   "image"_a, "vertices"_a, "color"_a,
                   "Draw a triangle with RGBA color and alpha blending (in-place)")
        .def_static("triangle_drawloss_batch_rgba",
                     &CPPRasterBackend::triangle_drawloss_batch_rgba,
                     "image"_a, "target_image"_a, "vertices"_a, "colors"_a, "out"_a,
                     "Draw a batch of triangles with RGBA colors and compute loss deltas");

    /*
    Stateful class managing GPU resources for CUDA-based rasterization.
    */
    nb::class_<CUDABackend>(m, "CUDABackend")
        .def(nb::init<
                 nb::ndarray<uint8_t, nb::shape<-1, -1, 3>, nb::c_contig, nb::device::cpu>,
                 nb::ndarray<uint8_t, nb::shape<-1, -1, 3>, nb::c_contig, nb::device::cpu>>(),
             nb::arg("background_image"),
             nb::arg("target_image"),
             "Constructor that initializes the backend with background and target images on the GPU.")
        .def("drawloss", &CUDABackend::drawloss,
             nb::arg("vertices"),
             nb::arg("colors"),
             nb::arg("out_losses"),
             "Calculates the loss for a batch of triangles against the target image.")
        .def("clone", &CUDABackend::clone,
             "Creates a deep copy of the current CUDABackend instance.");
}
