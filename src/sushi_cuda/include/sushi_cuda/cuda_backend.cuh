#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstdint>

namespace nb = nanobind;

struct ImageRGB
{
    uint8_t *data = nullptr;
    int width = 0;
    int height = 0;
};

/*
// --- CUDA Kernel ---
// This is a placeholder kernel. You'll replace this with your actual
// rasterization and loss calculation logic.
__global__ void drawloss_kernel(
    const ImageRGB background,
    const ImageRGB target,
    const int32_t *vertices,
    const uint8_t *colors,
    int64_t *losses,
    int num_triangles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_triangles)
    {
        return;
    }

    // TODO: Implement your triangle rasterization and loss calculation logic here.
    // This example just calculates a dummy loss value.

    // Example: Read the first vertex of the triangle
    int32_t v0_x = vertices[idx * 6 + 0];
    int32_t v0_y = vertices[idx * 6 + 1];

    // Example: Read the color for this triangle
    uint8_t r = colors[idx * 4 + 0];
    uint8_t g = colors[idx * 4 + 1];

    // Dummy loss calculation
    losses[idx] = static_cast<int64_t>(v0_x * r + v0_y * g);
}

void launch_drawloss_kernel(
    const ImageRGB &background,
    const ImageRGB &target,
    const int32_t *vertices,
    const uint8_t *colors,
    int64_t *losses,
    int num_triangles)
{
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_triangles + threads_per_block - 1) / threads_per_block;

    drawloss_kernel<<<blocks_per_grid, threads_per_block>>>(
        background, target, vertices, colors, losses, num_triangles);
}

*/

class CUDABackend
{
public:
    CUDABackend(
        nb::ndarray<uint8_t, nb::shape<-1, -1, 3>, nb::c_contig, nb::device::cpu> background_image,
        nb::ndarray<uint8_t, nb::shape<-1, -1, 3>, nb::c_contig, nb::device::cpu> target_image)
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

        // Allocate memory on the GPU
        cudaError_t err = cudaMalloc(&d_background_image.data, bg_size);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to allocate background image on GPU: " + std::string(cudaGetErrorString(err)));
        }

        // Copy data from host (numpy array) to device (GPU)
        err = cudaMemcpy(d_background_image.data, background_image.data(), bg_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cudaFree(d_background_image.data);
            throw std::runtime_error("Failed to copy background image to GPU: " + std::string(cudaGetErrorString(err)));
        }

        // --- Setup Target Image ---
        d_target_image.height = target_image.shape(0);
        d_target_image.width = target_image.shape(1);
        size_t target_size = d_target_image.height * d_target_image.width * 3 * sizeof(uint8_t);

        err = cudaMalloc(&d_target_image.data, target_size);
        if (err != cudaSuccess)
        {
            cudaFree(d_background_image.data); // Clean up previous allocation
            throw std::runtime_error("Failed to allocate target image on GPU: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaMemcpy(d_target_image.data, target_image.data(), target_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cudaFree(d_background_image.data);
            cudaFree(d_target_image.data);
            throw std::runtime_error("Failed to copy target image to GPU: " + std::string(cudaGetErrorString(err)));
        }
    }

    ~CUDABackend()
    {
        cudaFree(d_background_image.data);
        cudaFree(d_target_image.data);
    }

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

        cudaMalloc(&d_vertices, vertices.nbytes());
        cudaMalloc(&d_colors, colors.nbytes());
        cudaMalloc(&d_losses, out_losses.nbytes());

        // --- Copy inputs to GPU ---
        cudaMemcpy(d_vertices, vertices.data(), vertices.nbytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colors, colors.data(), colors.nbytes(), cudaMemcpyHostToDevice);

        // --- Launch CUDA Kernel ---
        // launch_drawloss_kernel(d_background_image, d_target_image, d_vertices, d_colors, d_losses, num_triangles);

        // --- Copy results back to host ---
        cudaMemcpy(out_losses.data(), d_losses, out_losses.nbytes(), cudaMemcpyDeviceToHost);

        // --- Free temporary GPU memory ---
        cudaFree(d_vertices);
        cudaFree(d_colors);
        cudaFree(d_losses);
    }

private:
    ImageRGB d_background_image;
    ImageRGB d_target_image;
};
