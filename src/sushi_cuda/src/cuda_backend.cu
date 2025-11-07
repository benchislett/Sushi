#include <sushi_cuda/cuda_utils.cuh>
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
    const uint32_t inv_a = 255 - fg_a;
    uint8_t mod_r = (uint8_t)(((uint32_t)fg_r * fg_a + (uint32_t)bg_r * inv_a + 128) >> 8);
    uint8_t mod_g = (uint8_t)(((uint32_t)fg_g * fg_a + (uint32_t)bg_g * inv_a + 128) >> 8);
    uint8_t mod_b = (uint8_t)(((uint32_t)fg_b * fg_a + (uint32_t)bg_b * inv_a + 128) >> 8);
    *out_r = (uint8_t) mod_r;
    *out_g = (uint8_t) mod_g;
    *out_b = (uint8_t) mod_b;
}

__global__ void drawloss_kernel_naive_triangle_parallel(
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

    int32_t min_x = v0_x;
    if (v1_x < min_x) min_x = v1_x;
    if (v2_x < min_x) min_x = v2_x;
    if (min_x < 0) min_x = 0;
    if (min_x >= background.width) min_x = background.width - 1;

    int32_t max_x = v0_x;
    if (v1_x > max_x) max_x = v1_x;
    if (v2_x > max_x) max_x = v2_x;
    if (max_x < 0) max_x = 0;
    if (max_x >= background.width) max_x = background.width - 1;

    int32_t min_y = v0_y;
    if (v1_y < min_y) min_y = v1_y;
    if (v2_y < min_y) min_y = v2_y;
    if (min_y < 0) min_y = 0;
    if (min_y >= background.height) min_y = background.height - 1;

    int32_t max_y = v0_y;
    if (v1_y > max_y) max_y = v1_y;
    if (v2_y > max_y) max_y = v2_y;
    if (max_y < 0) max_y = 0;
    if (max_y >= background.height) max_y = background.height - 1;

    int32_t w0_start = edge_function(v1_x, v1_y, v2_x, v2_y, min_x, min_y);
    int32_t w1_start = edge_function(v2_x, v2_y, v0_x, v0_y, min_x, min_y);
    int32_t w2_start = edge_function(v0_x, v0_y, v1_x, v1_y, min_x, min_y);

    int32_t w0_col_inc = (v2_y - v1_y);
    int32_t w1_col_inc = (v0_y - v2_y);
    int32_t w2_col_inc = (v1_y - v0_y);

    int32_t w0_row_inc = (v1_x - v2_x);
    int32_t w1_row_inc = (v2_x - v0_x);
    int32_t w2_row_inc = (v0_x - v1_x);

    int32_t w0_row = w0_start;
    int32_t w1_row = w1_start;
    int32_t w2_row = w2_start;
    for (int y = min_y; y <= max_y; ++y)
    {
        int32_t w0 = w0_row;
        int32_t w1 = w1_row;
        int32_t w2 = w2_row;
        for (int x = min_x; x <= max_x; ++x)
        {
            if (w0 <= 0 && w1 <= 0 && w2 <= 0)
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
            w0 += w0_col_inc;
            w1 += w1_col_inc;
            w2 += w2_col_inc;
        }
        w0_row += w0_row_inc;
        w1_row += w1_row_inc;
        w2_row += w2_row_inc;
    }
    losses[idx] = total_loss;
}

__global__ void drawloss_kernel_naive_pixel_parallel(
    const ImageRGB background,
    const ImageRGB target,
    const int32_t *vertices,
    const uint8_t *colors,
    int64_t *losses,
    int num_triangles)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= background.width || y >= background.height) {
        return;
    }

    const int pixel_index = y * background.width + x;

    int32_t target_r = (int32_t) target.data[pixel_index * 3 + 0];
    int32_t target_g = (int32_t) target.data[pixel_index * 3 + 1];
    int32_t target_b = (int32_t) target.data[pixel_index * 3 + 2];

    int32_t bg_r = (int32_t) background.data[pixel_index * 3 + 0];
    int32_t bg_g = (int32_t) background.data[pixel_index * 3 + 1];
    int32_t bg_b = (int32_t) background.data[pixel_index * 3 + 2];

    int32_t baseline_dr = bg_r - target_r;
    int32_t baseline_dg = bg_g - target_g;
    int32_t baseline_db = bg_b - target_b;
    int32_t baseline_loss = baseline_dr * baseline_dr + baseline_dg * baseline_dg + baseline_db * baseline_db;

    for (int i = 0; i < num_triangles; i++) {
        const int32_t v0_x = vertices[i * 6 + 0];
        const int32_t v0_y = vertices[i * 6 + 1];
        const int32_t v1_x = vertices[i * 6 + 2];
        const int32_t v1_y = vertices[i * 6 + 3];
        const int32_t v2_x = vertices[i * 6 + 4];
        const int32_t v2_y = vertices[i * 6 + 5];

        const uint8_t color_r = colors[i * 4 + 0];
        const uint8_t color_g = colors[i * 4 + 1];
        const uint8_t color_b = colors[i * 4 + 2];
        const uint8_t color_a = colors[i * 4 + 3];

        const int32_t w0 = edge_function(v1_x, v1_y, v2_x, v2_y, x, y);
        const int32_t w1 = edge_function(v2_x, v2_y, v0_x, v0_y, x, y);
        const int32_t w2 = edge_function(v0_x, v0_y, v1_x, v1_y, x, y);

        int32_t delta_loss = 0;
        if (w0 <= 0 && w1 <= 0 && w2 <= 0) {
            uint8_t out_r, out_g, out_b;
            composit_over(
                &out_r, &out_g, &out_b,
                color_r, color_g, color_b, color_a,
                bg_r, bg_g, bg_b
            );

            int32_t new_dr = (int32_t) out_r - target_r;
            int32_t new_dg = (int32_t) out_g - target_g;
            int32_t new_db = (int32_t) out_b - target_b;
            int32_t new_loss = new_dr * new_dr + new_dg * new_dg + new_db * new_db;

            delta_loss = new_loss - baseline_loss;
        }

        // Intra-warp reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            delta_loss += __shfl_down_sync(0xFFFFFFFF, delta_loss, offset);
        }

        // Global accumulation by warp leaders
        if (((threadIdx.x + threadIdx.y * blockDim.x) % 32) == 0) {
            atomicAdd((unsigned long long int*)&losses[i], (unsigned long long int)delta_loss);
        }
    }
}

void launch_drawloss_kernel_naive_triangle_parallel(
    const ImageRGB &background,
    const ImageRGB &target,
    const int32_t *vertices,
    const uint8_t *colors,
    int64_t *losses,
    int num_triangles) {
    constexpr int threads_per_block = 256;
    const int blocks_per_grid = (num_triangles + threads_per_block - 1) / threads_per_block;
    drawloss_kernel_naive_triangle_parallel<<<blocks_per_grid, threads_per_block>>>(
        background, target, vertices, colors, losses, num_triangles);
    CUDA_SYNC_CHECK();
}

void launch_drawloss_kernel_naive_pixel_parallel(
    const ImageRGB &background,
    const ImageRGB &target,
    const int32_t *vertices,
    const uint8_t *colors,
    int64_t *losses,
    int num_triangles) {
    constexpr int threads_per_block_x = 16;
    constexpr int threads_per_block_y = 16;
    const int blocks_per_grid_x = (background.width + threads_per_block_x - 1) / threads_per_block_x;
    const int blocks_per_grid_y = (background.height + threads_per_block_y - 1) / threads_per_block_y;
    dim3 blocks_per_grid(blocks_per_grid_x, blocks_per_grid_y);
    dim3 threads_per_block(threads_per_block_x, threads_per_block_y);
    drawloss_kernel_naive_pixel_parallel<<<blocks_per_grid, threads_per_block>>>(
        background, target, vertices, colors, losses, num_triangles);
    CUDA_SYNC_CHECK();
}
