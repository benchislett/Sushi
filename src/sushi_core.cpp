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
        /* FIRST IMPLEMENTATION: TILE-BASED RASTERIZATION WITH EDGE FUNCTIONS

        // Get array dimensions
        size_t height = image.shape(0);
        size_t width = image.shape(1);

        // Get direct pointers to the data
        uint8_t* image_data = image.data();
        const int32_t* vertex_data = vertices.data();
        const uint8_t* color_data = color.data();

        // Extract RGBA color components
        const uint8_t r = color_data[0];
        const uint8_t g = color_data[1];
        const uint8_t b = color_data[2];
        const uint8_t a = color_data[3];

        // If the color is fully transparent, there's nothing to do.
        if (a == 0) {
            return;
        }

        // Define vertices for clarity
        const int32_t p0_x = vertex_data[0], p0_y = vertex_data[1];
        const int32_t p1_x = vertex_data[2], p1_y = vertex_data[3];
        const int32_t p2_x = vertex_data[4], p2_y = vertex_data[5];

        // --- 1. Ensure Counter-Clockwise (CCW) Winding Order ---
        // We calculate the signed area of the triangle. A positive area indicates
        // CCW order, which our edge functions assume for "inside".
        // This formula MUST be consistent with the edge function formula below.
        int64_t signed_area = (int64_t)(p2_x - p0_x) * (p1_y - p0_y) - (int64_t)(p2_y - p0_y) * (p1_x - p0_x);

        // Degenerate or zero-area triangles have nothing to draw.
        if (signed_area == 0) {
            return;
        }

        const int32_t *v0, *v1, *v2;
        if (signed_area > 0) { // Already CCW
            v0 = &vertex_data[0]; v1 = &vertex_data[2]; v2 = &vertex_data[4];
        } else { // CW, swap v1 and v2 to make it CCW
            v0 = &vertex_data[0]; v1 = &vertex_data[4]; v2 = &vertex_data[2];
        }

        // --- 2. Calculate Bounding Box and Clip to Image ---
        int32_t min_x = std::max(0, std::min({v0[0], v1[0], v2[0]}));
        int32_t max_x = std::min((int32_t)width, std::max({v0[0], v1[0], v2[0]}) + 1);
        int32_t min_y = std::max(0, std::min({v0[1], v1[1], v2[1]}));
        int32_t max_y = std::min((int32_t)height, std::max({v0[1], v1[1], v2[1]}) + 1);

        if (min_x >= max_x || min_y >= max_y) {
            return; // Triangle is completely off-screen.
        }

        // --- 3. Setup Tiling and Edge Functions ---
        const int TILE_SIZE = 16;
        int32_t tile_min_x = min_x & ~(TILE_SIZE - 1);
        int32_t tile_min_y = min_y & ~(TILE_SIZE - 1);

        // Edge function: E(x, y) = (x-vx)*dy - (y-vy)*dx.
        // A point is inside if E >= 0 for all three edges (due to CCW order).
        const int64_t dx01 = v1[0] - v0[0]; const int64_t dy01 = v1[1] - v0[1];
        const int64_t dx12 = v2[0] - v1[0]; const int64_t dy12 = v2[1] - v1[1];
        const int64_t dx20 = v0[0] - v2[0]; const int64_t dy20 = v0[1] - v2[1];

        // --- 4. Iterate Over Tiles ---
        for (int32_t ty = tile_min_y; ty < max_y; ty += TILE_SIZE) {
            for (int32_t tx = tile_min_x; tx < max_x; tx += TILE_SIZE) {
                // Evaluate edge functions at the four corners of the tile.
                int64_t w0_c00 = (int64_t)(tx - v0[0]) * dy01 - (int64_t)(ty - v0[1]) * dx01;
                int64_t w1_c00 = (int64_t)(tx - v1[0]) * dy12 - (int64_t)(ty - v1[1]) * dx12;
                int64_t w2_c00 = (int64_t)(tx - v2[0]) * dy20 - (int64_t)(ty - v2[1]) * dx20;

                // If all four corners are outside any single edge, the entire tile is outside.
                if ( (std::max({w0_c00, w0_c00 + (TILE_SIZE-1)*dy01, w0_c00 - (TILE_SIZE-1)*dx01, w0_c00 + (TILE_SIZE-1)*dy01 - (TILE_SIZE-1)*dx01}) < 0) ||
                     (std::max({w1_c00, w1_c00 + (TILE_SIZE-1)*dy12, w1_c00 - (TILE_SIZE-1)*dx12, w1_c00 + (TILE_SIZE-1)*dy12 - (TILE_SIZE-1)*dx12}) < 0) ||
                     (std::max({w2_c00, w2_c00 + (TILE_SIZE-1)*dy20, w2_c00 - (TILE_SIZE-1)*dx20, w2_c00 + (TILE_SIZE-1)*dy20 - (TILE_SIZE-1)*dx20}) < 0) ) {
                    continue; // Skip this tile.
                }

                // --- 5. Iterate Over Pixels Within the Tile ---
                int32_t y_start = std::max(ty, min_y);
                int32_t y_end = std::min(ty + TILE_SIZE, max_y);
                int32_t x_start = std::max(tx, min_x);
                int32_t x_end = std::min(tx + TILE_SIZE, max_x);

                int64_t w0_row = (int64_t)(x_start - v0[0]) * dy01 - (int64_t)(y_start - v0[1]) * dx01;
                int64_t w1_row = (int64_t)(x_start - v1[0]) * dy12 - (int64_t)(y_start - v1[1]) * dx12;
                int64_t w2_row = (int64_t)(x_start - v2[0]) * dy20 - (int64_t)(y_start - v2[1]) * dx20;

                for (int32_t y = y_start; y < y_end; ++y) {
                    int64_t w0 = w0_row;
                    int64_t w1 = w1_row;
                    int64_t w2 = w2_row;
                    uint8_t* pixel = image_data + (y * width + x_start) * 3;

                    for (int32_t x = x_start; x < x_end; ++x) {
                        // If all edge function values are non-negative, the pixel is inside.
                        if ((w0 | w1 | w2) >= 0) {
                            // Opaque case: just overwrite the pixel color.
                            if (a == 255) {
                                pixel[0] = r; pixel[1] = g; pixel[2] = b;
                            } else { // Alpha blending case.
                                const uint32_t inv_a = 255 - a;
                                pixel[0] = (uint8_t)(((uint32_t)r * a + (uint32_t)pixel[0] * inv_a + 255) >> 8);
                                pixel[1] = (uint8_t)(((uint32_t)g * a + (uint32_t)pixel[1] * inv_a + 255) >> 8);
                                pixel[2] = (uint8_t)(((uint32_t)b * a + (uint32_t)pixel[2] * inv_a + 255) >> 8);
                            }
                        }
                        // Incrementally update for the next pixel in the row.
                        w0 += dy01; w1 += dy12; w2 += dy20;
                        pixel += 3;
                    }
                    // Incrementally update for the start of the next row.
                    w0_row -= dx01; w1_row -= dx12; w2_row -= dx20;
                }
            }
        }
        */

        /* SECOND IMPLEMENTATION: SCANLINE RASTERIZATION */
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
        nb::ndarray<nb::numpy, float, nb::shape<-1> > out
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

        float* out_data = out.data();

        const double normalizer = 1.0 / (double)(width * height * 3);

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
                out_data[i] = 0.0f;
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

            out_data[i] = (float)(total_sq_err_delta * normalizer);
        }
    }
};



NB_MODULE(sushi_core, m) {
    m.doc() = "Sushi C++ rasterization backend";

    nb::class_<CPPRasterBackend>(m, "CPPRasterBackend")
        .def_static("triangle_draw_single_rgba_inplace",
                   &CPPRasterBackend::triangle_draw_single_rgba_inplace,
                   "image"_a, "vertices"_a, "color"_a,
                   "Draw a triangle with RGBA color and alpha blending (in-place)")
        .def_static("triangle_drawloss_batch_rgba",
                     &CPPRasterBackend::triangle_drawloss_batch_rgba,
                     "image"_a, "target_image"_a, "vertices"_a, "colors"_a, "out"_a,
                     "Draw a batch of triangles with RGBA colors and compute loss deltas");
}
