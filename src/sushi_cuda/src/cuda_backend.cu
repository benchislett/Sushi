#include <sushi_cuda/cuda_backend.cuh>

__host__ __device__ int32_t edge_function(
    const int32_t ax, const int32_t ay,
    const int32_t bx, const int32_t by,
    const int32_t cx, const int32_t cy)
{
    return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax);
}

__host__ __device__ void composit_over(
    uint8_t *out_r, uint8_t *out_g, uint8_t *out_b,
    uint8_t fg_r, uint8_t fg_g, uint8_t fg_b, uint8_t fg_a,
    uint8_t bg_r, uint8_t bg_g, uint8_t bg_b)
{
    float alpha = fg_a / 255.0f;
    *out_r = static_cast<uint8_t>(fg_r * alpha + bg_r * (1.0f - alpha));
    *out_g = static_cast<uint8_t>(fg_g * alpha + bg_g * (1.0f - alpha));
    *out_b = static_cast<uint8_t>(fg_b * alpha + bg_b * (1.0f - alpha));
    // uint8_t alpha = fg_a + ((bg_r | bg_g | bg_b) != 0 ? (255 - fg_a) : 0);
    // if (alpha == 0)
    // {
    //     *out_r = 0;
    //     *out_g = 0;
    //     *out_b = 0;
    // }
    // else
    // {
    //     *out_r = (fg_r * fg_a + bg_r * (255 - fg_a)) / alpha;
    //     *out_g = (fg_g * fg_a + bg_g * (255 - fg_a)) / alpha;
    //     *out_b = (fg_b * fg_a + bg_b * (255 - fg_a)) / alpha;
    // }
}

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

    const int32_t v0_x = vertices[idx * 6 + 0];
    const int32_t v0_y = vertices[idx * 6 + 1];
    const int32_t v1_x = vertices[idx * 6 + 2];
    const int32_t v1_y = vertices[idx * 6 + 3];
    const int32_t v2_x = vertices[idx * 6 + 4];
    const int32_t v2_y = vertices[idx * 6 + 5];

    const uint8_t color_r = colors[idx * 4 + 0];
    const uint8_t color_g = colors[idx * 4 + 1];
    const uint8_t color_b = colors[idx * 4 + 2];
    const uint8_t color_a = colors[idx * 4 + 3];

    int64_t total_loss = 0;

    for (int y = 0; y < background.height; ++y)
    {
        for (int x = 0; x < background.width; ++x)
        {
            const int32_t w0 = edge_function(v1_x, v1_y, v2_x, v2_y, x, y);
            const int32_t w1 = edge_function(v2_x, v2_y, v0_x, v0_y, x, y);
            const int32_t w2 = edge_function(v0_x, v0_y, v1_x, v1_y, x, y);

            if (w0 >= 0 && w1 >= 0 && w2 >= 0)
            {
                const int bg_index = (y * background.width + x) * 3;
                const int tgt_index = (y * target.width + x) * 3;

                uint8_t out_r, out_g, out_b;
                composit_over(
                    &out_r, &out_g, &out_b,
                    color_r, color_g, color_b, color_a,
                    background.data[bg_index], background.data[bg_index + 1], background.data[bg_index + 2]
                );

                int32_t baseline_dr = (int32_t) background.data[bg_index + 0] - (int32_t) target.data[tgt_index + 0];
                int32_t baseline_dg = (int32_t) background.data[bg_index + 1] - (int32_t) target.data[tgt_index + 1];
                int32_t baseline_db = (int32_t) background.data[bg_index + 2] - (int32_t) target.data[tgt_index + 2];
                int32_t baseline_loss = baseline_dr * baseline_dr + baseline_dg * baseline_dg + baseline_db * baseline_db;

                int32_t new_dr = (int32_t) out_r - (int32_t) target.data[tgt_index + 0];
                int32_t new_dg = (int32_t) out_g - (int32_t) target.data[tgt_index + 1];
                int32_t new_db = (int32_t) out_b - (int32_t) target.data[tgt_index + 2];
                int32_t new_loss = new_dr * new_dr + new_dg * new_dg + new_db * new_db;

                total_loss += (new_loss - baseline_loss);
            }
        }
    }
    losses[idx] = total_loss;
}

void launch_drawloss_kernel(
    const ImageRGB &background,
    const ImageRGB &target,
    const int32_t *vertices,
    const uint8_t *colors,
    int64_t *losses,
    int num_triangles) {
    constexpr int threads_per_block = 256;
    const int blocks_per_grid = (num_triangles + threads_per_block - 1) / threads_per_block;
    drawloss_kernel<<<blocks_per_grid, threads_per_block>>>(
        background, target, vertices, colors, losses, num_triangles);
    cudaDeviceSynchronize();
}
