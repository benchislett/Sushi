#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include <sushi_cuda/cuda_utils.cuh>

#include <cstdint>
#include <string>
#include <iostream>
#include <stdexcept>
#include <memory>

namespace nb = nanobind;

void launch_drawloss_kernel_naive_triangle_parallel(
    const ImageRGB &background,
    const ImageRGB &target,
    const int32_t *vertices,
    const uint8_t *colors,
    int64_t *losses,
    int num_triangles);

void launch_drawloss_kernel_naive_pixel_parallel(
    const ImageRGB &background,
    const ImageRGB &target,
    const int32_t *vertices,
    const uint8_t *colors,
    int64_t *losses,
    int num_triangles);

class CUDABackend
{
public:
    CUDABackend(
        nb::ndarray<uint8_t, nb::shape<-1, -1, 3>, nb::c_contig, nb::device::cpu> background_image,
        nb::ndarray<uint8_t, nb::shape<-1, -1, 3>, nb::c_contig, nb::device::cpu> target_image,
        std::string method
    ) : m_method(std::move(method))
    {
        if (background_image.ndim() != 3 || target_image.ndim() != 3)
        {
            throw std::runtime_error("Background and target images must be 3D arrays.");
        }
        if (background_image.shape(0) != target_image.shape(0) || background_image.shape(1) != target_image.shape(1))
        {
            throw std::runtime_error("Background and target images must have the same height and width.");
        }

        // --- Setup Background Image ---
        d_background_image.height = background_image.shape(0);
        d_background_image.width = background_image.shape(1);
        size_t bg_size = d_background_image.height * d_background_image.width * 3 * sizeof(uint8_t);

        CUDA_CHECK_LAST_ERROR();

        // Allocate memory on the GPU
        CUDA_CHECK(cudaMalloc(&d_background_image.data, bg_size));

        // Copy data from host (numpy array) to device (GPU)
        CUDA_CHECK(cudaMemcpy(d_background_image.data, background_image.data(), bg_size, cudaMemcpyHostToDevice));

        // --- Setup Target Image ---
        d_target_image.height = target_image.shape(0);
        d_target_image.width = target_image.shape(1);
        size_t target_size = d_target_image.height * d_target_image.width * 3 * sizeof(uint8_t);

        CUDA_CHECK(cudaMalloc(&d_target_image.data, target_size));
        CUDA_CHECK(cudaMemcpy(d_target_image.data, target_image.data(), target_size, cudaMemcpyHostToDevice));
    }

    CUDABackend(ImageRGB background_image, ImageRGB target_image, std::string method)
        : d_background_image(background_image), d_target_image(target_image), m_method(std::move(method)) {}

    ~CUDABackend() noexcept(false)
    {
        if (d_background_image.data)
        {
            CUDA_CHECK(cudaFree(d_background_image.data));
        }
        if (d_target_image.data)
        {
            CUDA_CHECK(cudaFree(d_target_image.data));
        }
    }

    CUDABackend(CUDABackend&& other) noexcept
        : d_background_image(other.d_background_image),
          d_target_image(other.d_target_image),
          m_method(std::move(other.m_method))
    {
        other.d_background_image.data = nullptr;
        other.d_target_image.data = nullptr;
    }

    CUDABackend(const CUDABackend&) = delete; // Disable copy constructor
    CUDABackend& operator=(const CUDABackend&) = delete; // Disable copy assignment
    CUDABackend& operator=(CUDABackend&&) = delete; // Disable move assignment

    void drawloss(
        nb::ndarray<int32_t, nb::shape<-1, 3, 2>, nb::c_contig, nb::device::cpu> vertices,
        nb::ndarray<uint8_t, nb::shape<-1, 4>, nb::c_contig, nb::device::cpu> colors,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig, nb::device::cpu> out_losses)
    {
        if (vertices.shape(0) != colors.shape(0) || vertices.shape(0) != out_losses.shape(0))
        {
            throw std::runtime_error("Input arrays (vertices, colors, out_losses) must have the same batch size.");
        }
        const size_t num_triangles = vertices.shape(0);
        if (num_triangles == 0)
            return;

        // --- Allocate GPU memory for inputs/outputs ---
        int32_t *d_vertices;
        uint8_t *d_colors;
        int64_t *d_losses;


        CUDA_CHECK_LAST_ERROR();

        CUDA_CHECK(cudaMalloc(&d_vertices, vertices.nbytes()));
        CUDA_CHECK(cudaMalloc(&d_colors, colors.nbytes()));
        CUDA_CHECK(cudaMalloc(&d_losses, out_losses.nbytes()));

        // --- Copy inputs to GPU ---
        CUDA_CHECK(cudaMemcpy(d_vertices, vertices.data(), vertices.nbytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colors, colors.data(), colors.nbytes(), cudaMemcpyHostToDevice));
        // Zero out the losses array on the GPU
        CUDA_CHECK(cudaMemset(d_losses, 0, out_losses.nbytes()));

        // --- Launch CUDA Kernel ---
        if (m_method == "naive-triangle-parallel")
        {
            launch_drawloss_kernel_naive_triangle_parallel(
                d_background_image, d_target_image, d_vertices, d_colors, d_losses, num_triangles);
        }
        else if (m_method == "naive-pixel-parallel")
        {
            launch_drawloss_kernel_naive_pixel_parallel(
                d_background_image, d_target_image, d_vertices, d_colors, d_losses, num_triangles);
        } else {
            throw std::runtime_error("Unknown method: " + m_method);
        }

        // --- Copy results back to host ---
        CUDA_CHECK(cudaMemcpy(out_losses.data(), d_losses, out_losses.nbytes(), cudaMemcpyDeviceToHost));
        CUDA_SYNC_CHECK();

        // --- Free temporary GPU memory ---
        CUDA_CHECK(cudaFree(d_vertices));
        CUDA_CHECK(cudaFree(d_colors));
        CUDA_CHECK(cudaFree(d_losses));

        CUDA_CHECK_LAST_ERROR();
    }

    std::unique_ptr<CUDABackend> clone() const
    {
        ImageRGB d_bg_copy;
        ImageRGB d_target_copy;

        size_t bg_size = d_background_image.height * d_background_image.width * 3 * sizeof(uint8_t);
        size_t target_size = d_target_image.height * d_target_image.width * 3 * sizeof(uint8_t);

        CUDA_CHECK_LAST_ERROR();

        CUDA_CHECK(cudaMalloc(&d_bg_copy.data, bg_size));
        CUDA_CHECK(cudaMemcpy(d_bg_copy.data, d_background_image.data, bg_size, cudaMemcpyDeviceToDevice));
        d_bg_copy.height = d_background_image.height;
        d_bg_copy.width = d_background_image.width;

        CUDA_CHECK(cudaMalloc(&d_target_copy.data, target_size));
        CUDA_CHECK(cudaMemcpy(d_target_copy.data, d_target_image.data, target_size, cudaMemcpyDeviceToDevice));
        d_target_copy.height = d_target_image.height;
        d_target_copy.width = d_target_image.width;

        CUDA_SYNC_CHECK();

        return std::make_unique<CUDABackend>(d_bg_copy, d_target_copy, m_method);
    }

private:
    ImageRGB d_background_image;
    ImageRGB d_target_image;
    std::string m_method;
};
