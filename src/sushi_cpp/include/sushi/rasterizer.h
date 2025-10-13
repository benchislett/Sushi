#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <array>
#include <optional>

namespace nb = nanobind;
using namespace nb::literals;

class CPPRasterBackend {
public:
    // Triangle rasterization with RGBA color and alpha blending
    static void triangle_draw_single_rgba_inplace(
        nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, 3>, nb::c_contig> image,
        nb::ndarray<nb::numpy, int32_t, nb::shape<3, 2>, nb::c_contig> vertices,
        nb::ndarray<nb::numpy, uint8_t, nb::shape<4>, nb::c_contig> color
    ) {
        // Get array dimensions
        const int32_t height = image.shape(0);
        const int32_t width = image.shape(1);

        // Get direct pointers to the data
        uint8_t* image_data = image.data();
        const int32_t* vertex_data = vertices.data();
        const uint8_t* color_data = color.data();

        // Extract RGBA color components
        const uint8_t r = color_data[0];
        const uint8_t g = color_data[1];
        const uint8_t b = color_data[2];
        const uint8_t a = color_data[3];

        if (a == 0) {
            return; // Fully transparent, nothing to do.
        }

        // --- 1. Sort Vertices by Y-coordinate ---
        // Sort v0, v1, v2 so that v0.y <= v1.y <= v2.y
        const int32_t* v_ptr[3] = {&vertex_data[0], &vertex_data[2], &vertex_data[4]};
        if (v_ptr[0][1] > v_ptr[1][1]) std::swap(v_ptr[0], v_ptr[1]);
        if (v_ptr[1][1] > v_ptr[2][1]) std::swap(v_ptr[1], v_ptr[2]);
        if (v_ptr[0][1] > v_ptr[1][1]) std::swap(v_ptr[0], v_ptr[1]);

        const int32_t *v0 = v_ptr[0], *v1 = v_ptr[1], *v2 = v_ptr[2];

        // --- 2. Render Triangle Halves ---
        // The triangle is split at v1. We render the top part (v0-v1) and
        // bottom part (v1-v2) separately.

        // Fixed-point precision (16 bits for fractional part)
        const int32_t FIXED_SHIFT = 16;

        // Draw the top part of the triangle (bottom-flat)
        if (v0[1] < v1[1]) {
            int32_t dy01 = v1[1] - v0[1];
            int32_t dy02 = v2[1] - v0[1];

            // Use 64-bit integers for fixed-point calculations to prevent overflow
            int64_t dx01_fixed = ((int64_t)(v1[0] - v0[0]) << FIXED_SHIFT) / dy01;
            int64_t dx02_fixed = dy02 == 0 ? 0 : ((int64_t)(v2[0] - v0[0]) << FIXED_SHIFT) / dy02;

            int64_t x1_fixed = (int64_t)v0[0] << FIXED_SHIFT;
            int64_t x2_fixed = (int64_t)v0[0] << FIXED_SHIFT;

            int32_t y_start = std::max(0, v0[1]);
            int32_t y_end = std::min(height, v1[1]);

            // Fast-forward x positions to the first visible scanline
            if (v0[1] < y_start) {
                int32_t y_diff = y_start - v0[1];
                x1_fixed += dx01_fixed * y_diff;
                x2_fixed += dx02_fixed * y_diff;
            }

            for (int32_t y = y_start; y < y_end; ++y) {
                int32_t x_start = x1_fixed < x2_fixed ? (x1_fixed >> FIXED_SHIFT) : (x2_fixed >> FIXED_SHIFT);
                int32_t x_end = x1_fixed < x2_fixed ? (x2_fixed >> FIXED_SHIFT) : (x1_fixed >> FIXED_SHIFT);

                int32_t x_start_clipped = std::max(0, x_start);
                int32_t x_end_clipped = std::min(width, x_end + 1);

                uint8_t* pixel = image_data + (y * width + x_start_clipped) * 3;
                for (int32_t x = x_start_clipped; x < x_end_clipped; ++x) {
                    if (a == 255) {
                        pixel[0] = r; pixel[1] = g; pixel[2] = b;
                    } else {
                        const uint32_t inv_a = 255 - a;
                        pixel[0] = (uint8_t)(((uint32_t)r * a + (uint32_t)pixel[0] * inv_a + 128) >> 8);
                        pixel[1] = (uint8_t)(((uint32_t)g * a + (uint32_t)pixel[1] * inv_a + 128) >> 8);
                        pixel[2] = (uint8_t)(((uint32_t)b * a + (uint32_t)pixel[2] * inv_a + 128) >> 8);
                    }
                    pixel += 3;
                }
                x1_fixed += dx01_fixed;
                x2_fixed += dx02_fixed;
            }
        }

        // Draw the bottom part of the triangle (top-flat)
        if (v1[1] < v2[1]) {
            int32_t dy12 = v2[1] - v1[1];
            int32_t dy02 = v2[1] - v0[1];

            int64_t dx12_fixed = ((int64_t)(v2[0] - v1[0]) << FIXED_SHIFT) / dy12;
            int64_t dx02_fixed = dy02 == 0 ? 0 : ((int64_t)(v2[0] - v0[0]) << FIXED_SHIFT) / dy02;

            int64_t x1_fixed = (int64_t)v1[0] << FIXED_SHIFT;
            int64_t x2_fixed = (int64_t)v0[0] << FIXED_SHIFT;
            if (dy02 != 0) { // Calculate where the v0-v2 edge is at the midpoint y
                 x2_fixed += ((int64_t)(v2[0] - v0[0]) << FIXED_SHIFT) * (v1[1] - v0[1]) / dy02;
            }

            int32_t y_start = std::max(0, v1[1]);
            int32_t y_end = std::min(height, v2[1]);

            // Fast-forward x positions to the first visible scanline
            if (v1[1] < y_start) {
                int32_t y_diff = y_start - v1[1];
                x1_fixed += dx12_fixed * y_diff;
                x2_fixed += dx02_fixed * y_diff;
            }

            for (int32_t y = y_start; y < y_end; ++y) {
                int32_t x_start = x1_fixed < x2_fixed ? (x1_fixed >> FIXED_SHIFT) : (x2_fixed >> FIXED_SHIFT);
                int32_t x_end = x1_fixed < x2_fixed ? (x2_fixed >> FIXED_SHIFT) : (x1_fixed >> FIXED_SHIFT);

                int32_t x_start_clipped = std::max(0, x_start);
                int32_t x_end_clipped = std::min(width, x_end + 1);

                uint8_t* pixel = image_data + (y * width + x_start_clipped) * 3;
                for (int32_t x = x_start_clipped; x < x_end_clipped; ++x) {
                    if (a == 255) {
                        pixel[0] = r; pixel[1] = g; pixel[2] = b;
                    } else {
                        const uint32_t inv_a = 255 - a;
                        pixel[0] = (uint8_t)(((uint32_t)r * a + (uint32_t)pixel[0] * inv_a + 128) >> 8);
                        pixel[1] = (uint8_t)(((uint32_t)g * a + (uint32_t)pixel[1] * inv_a + 128) >> 8);
                        pixel[2] = (uint8_t)(((uint32_t)b * a + (uint32_t)pixel[2] * inv_a + 128) >> 8);
                    }
                    pixel += 3;
                }
                x1_fixed += dx12_fixed;
                x2_fixed += dx02_fixed;
            }
        }
    }

    static void
    triangle_drawloss_batch_rgba(
        nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, 3>, nb::c_contig> image,
        nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, 3>, nb::c_contig> target_image,
        nb::ndarray<nb::numpy, int32_t, nb::shape<-1, 3, 2>, nb::c_contig> vertices,
        nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, 4>, nb::c_contig> colors,
        nb::ndarray<nb::numpy, int64_t, nb::shape<-1>, nb::c_contig> out
    ) {
        // Get image dimensions
        const int32_t height = image.shape(0);
        const int32_t width = image.shape(1);
        const size_t num_triangles = vertices.shape(0);

        // Get direct pointers to data
        const uint8_t* image_data = image.data();
        const uint8_t* target_image_data = target_image.data();
        const int32_t* vertices_data = vertices.data();
        const uint8_t* colors_data = colors.data();

        int64_t* out_data = out.data();

        // --- Main loop over the batch of triangles ---
        for (size_t i = 0; i < num_triangles; ++i) {
            const int32_t* current_vertices = vertices_data + i * 6;
            const uint8_t* current_color = colors_data + i * 4;

            const uint8_t r = current_color[0];
            const uint8_t g = current_color[1];
            const uint8_t b = current_color[2];
            const uint8_t a = current_color[3];

            // If triangle is transparent, it causes no change in loss.
            if (a == 0) {
                out_data[i] = 0;
                continue;
            }

            // --- 1. Sort Vertices by Y-coordinate ---
            const int32_t* v_ptr[3] = {&current_vertices[0], &current_vertices[2], &current_vertices[4]};
            if (v_ptr[0][1] > v_ptr[1][1]) std::swap(v_ptr[0], v_ptr[1]);
            if (v_ptr[1][1] > v_ptr[2][1]) std::swap(v_ptr[1], v_ptr[2]);
            if (v_ptr[0][1] > v_ptr[1][1]) std::swap(v_ptr[0], v_ptr[1]);

            const int32_t *v0 = v_ptr[0], *v1 = v_ptr[1], *v2 = v_ptr[2];

            // This accumulator holds the sum of (new_error^2 - old_error^2)
            int64_t total_sq_err_delta = 0;
            const int32_t FIXED_SHIFT = 16;

            // --- 2. Rasterize top half (bottom-flat triangle) ---
            if (v0[1] < v1[1]) {
                int32_t dy01 = v1[1] - v0[1];
                int32_t dy02 = v2[1] - v0[1];

                int64_t dx01_fixed = ((int64_t)(v1[0] - v0[0]) << FIXED_SHIFT) / dy01;
                int64_t dx02_fixed = dy02 == 0 ? 0 : ((int64_t)(v2[0] - v0[0]) << FIXED_SHIFT) / dy02;

                int64_t x1_fixed = (int64_t)v0[0] << FIXED_SHIFT;
                int64_t x2_fixed = (int64_t)v0[0] << FIXED_SHIFT;

                int32_t y_start = std::max(0, v0[1]);
                int32_t y_end = std::min(height, v1[1]);

                if (v0[1] < y_start) {
                    int32_t y_diff = y_start - v0[1];
                    x1_fixed += dx01_fixed * y_diff;
                    x2_fixed += dx02_fixed * y_diff;
                }

                for (int32_t y = y_start; y < y_end; ++y) {
                    int32_t x_start = x1_fixed < x2_fixed ? (x1_fixed >> FIXED_SHIFT) : (x2_fixed >> FIXED_SHIFT);
                    int32_t x_end = x1_fixed < x2_fixed ? (x2_fixed >> FIXED_SHIFT) : (x1_fixed >> FIXED_SHIFT);

                    int32_t x_start_clipped = std::max(0, x_start);
                    int32_t x_end_clipped = std::min(width, x_end + 1);

                    for (int32_t x = x_start_clipped; x < x_end_clipped; ++x) {
                        const size_t idx = (y * width + x) * 3;
                        const uint8_t *p_img = image_data + idx;
                        const uint8_t *p_tgt = target_image_data + idx;

                        int32_t err_old_r = (int32_t)p_img[0] - p_tgt[0];
                        int32_t err_old_g = (int32_t)p_img[1] - p_tgt[1];
                        int32_t err_old_b = (int32_t)p_img[2] - p_tgt[2];

                        const uint32_t inv_a = 255 - a;
                        uint8_t mod_r = (uint8_t)(((uint32_t)r * a + (uint32_t)p_img[0] * inv_a + 128) >> 8);
                        uint8_t mod_g = (uint8_t)(((uint32_t)g * a + (uint32_t)p_img[1] * inv_a + 128) >> 8);
                        uint8_t mod_b = (uint8_t)(((uint32_t)b * a + (uint32_t)p_img[2] * inv_a + 128) >> 8);

                        int32_t err_new_r = (int32_t)mod_r - p_tgt[0];
                        int32_t err_new_g = (int32_t)mod_g - p_tgt[1];
                        int32_t err_new_b = (int32_t)mod_b - p_tgt[2];

                        total_sq_err_delta += (err_new_r*err_new_r - err_old_r*err_old_r) +
                                              (err_new_g*err_new_g - err_old_g*err_old_g) +
                                              (err_new_b*err_new_b - err_old_b*err_old_b);
                    }
                    x1_fixed += dx01_fixed;
                    x2_fixed += dx02_fixed;
                }
            }

            // --- 3. Rasterize bottom half (top-flat triangle) ---
            if (v1[1] < v2[1]) {
                int32_t dy12 = v2[1] - v1[1];
                int32_t dy02 = v2[1] - v0[1];

                int64_t dx12_fixed = ((int64_t)(v2[0] - v1[0]) << FIXED_SHIFT) / dy12;
                int64_t dx02_fixed = dy02 == 0 ? 0 : ((int64_t)(v2[0] - v0[0]) << FIXED_SHIFT) / dy02;

                int64_t x1_fixed = (int64_t)v1[0] << FIXED_SHIFT;
                int64_t x2_fixed = (int64_t)v0[0] << FIXED_SHIFT;
                if (dy02 != 0) {
                     x2_fixed += ((int64_t)(v2[0] - v0[0]) << FIXED_SHIFT) * (v1[1] - v0[1]) / dy02;
                }

                int32_t y_start = std::max(0, v1[1]);
                int32_t y_end = std::min(height, v2[1]);

                if (v1[1] < y_start) {
                    int32_t y_diff = y_start - v1[1];
                    x1_fixed += dx12_fixed * y_diff;
                    x2_fixed += dx02_fixed * y_diff;
                }

                for (int32_t y = y_start; y < y_end; ++y) {
                    int32_t x_start = x1_fixed < x2_fixed ? (x1_fixed >> FIXED_SHIFT) : (x2_fixed >> FIXED_SHIFT);
                    int32_t x_end = x1_fixed < x2_fixed ? (x2_fixed >> FIXED_SHIFT) : (x1_fixed >> FIXED_SHIFT);

                    int32_t x_start_clipped = std::max(0, x_start);
                    int32_t x_end_clipped = std::min(width, x_end + 1);

                    for (int32_t x = x_start_clipped; x < x_end_clipped; ++x) {
                        const size_t idx = (y * width + x) * 3;
                        const uint8_t *p_img = image_data + idx;
                        const uint8_t *p_tgt = target_image_data + idx;

                        int32_t err_old_r = (int32_t)p_img[0] - p_tgt[0];
                        int32_t err_old_g = (int32_t)p_img[1] - p_tgt[1];
                        int32_t err_old_b = (int32_t)p_img[2] - p_tgt[2];

                        const uint32_t inv_a = 255 - a;
                        uint8_t mod_r = (uint8_t)(((uint32_t)r * a + (uint32_t)p_img[0] * inv_a + 128) >> 8);
                        uint8_t mod_g = (uint8_t)(((uint32_t)g * a + (uint32_t)p_img[1] * inv_a + 128) >> 8);
                        uint8_t mod_b = (uint8_t)(((uint32_t)b * a + (uint32_t)p_img[2] * inv_a + 128) >> 8);

                        int32_t err_new_r = (int32_t)mod_r - p_tgt[0];
                        int32_t err_new_g = (int32_t)mod_g - p_tgt[1];
                        int32_t err_new_b = (int32_t)mod_b - p_tgt[2];

                        total_sq_err_delta += (err_new_r*err_new_r - err_old_r*err_old_r) +
                                              (err_new_g*err_new_g - err_old_g*err_old_g) +
                                              (err_new_b*err_new_b - err_old_b*err_old_b);
                    }
                    x1_fixed += dx12_fixed;
                    x2_fixed += dx02_fixed;
                }
            }

            out_data[i] = total_sq_err_delta;
        }
    }
};
