#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>

// --- Configuration ---

// Number of triangles to rasterize. You can change this value.
constexpr int N = 256;
// Tile dimensions. The paper uses 32x32 bins.
constexpr int TILE_DIM = 32;
// Number of untimed warmup runs to perform before the measured run.
constexpr int WARMUP_RUNS = 3;

// --- Optional Reduction Strategy for the Reference Kernel ---
// Uncomment exactly ONE of the following three lines to choose a strategy.
// 1. USE_GLOBAL_ATOMIC:         Each thread uses atomicAdd directly on global memory. Simple but high contention.
// 2. USE_WARP_REDUCE_TO_GLOBAL: Threads in a warp sum their results, and only one thread per
//                               warp performs a global atomicAdd. Reduces contention by 32x.
// 3. USE_SHMEM_ATOMIC_THEN_COPY: Each thread uses atomicAdd on a fast shared memory array. The results
//                                are then copied to global memory in a final step.

// #define USE_GLOBAL_ATOMIC
// #define USE_WARP_REDUCE_TO_GLOBAL
#define USE_SHMEM_ATOMIC_THEN_COPY

// --- Sanity check to ensure only one strategy is chosen ---
#if (defined(USE_GLOBAL_ATOMIC) + defined(USE_WARP_REDUCE_TO_GLOBAL) + defined(USE_SHMEM_ATOMIC_THEN_COPY)) != 1
#error "Please uncomment exactly one reduction strategy macro."
#endif


// --- Helper Functions and Structs ---

// struct int2 { int x, y; };
struct Triangle { int2 v0, v1, v2; };

#define CUDA_CHECK(err) { \
    cudaError_t e = err; \
    if (e != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// --- Device Code ---

__device__ inline int edge_function(int2 v0, int2 v1, int2 p) {
    return (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x);
}

/**
 * @brief Reference Kernel ("Dumb" Method) with Preprocessor-Switched Strategy
 */
__global__ void reference_raster_kernel(const Triangle* triangles, int* out_pixel_counts) {
    // --- Shared memory setup for the ShmemAtomicThenCopy strategy ---
    #if defined(USE_SHMEM_ATOMIC_THEN_COPY)
        extern __shared__ int shmem_counts[];
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            shmem_counts[i] = 0;
        }
        __syncthreads();
    #endif

    // --- Main Pixel Processing Loop ---
    int thread_id = threadIdx.x;
    int2 p = { thread_id % TILE_DIM, thread_id / TILE_DIM };

    for (int i = 0; i < N; ++i) {
        Triangle t = triangles[i];
        int e0 = edge_function(t.v0, t.v1, p);
        int e1 = edge_function(t.v1, t.v2, p);
        int e2 = edge_function(t.v2, t.v0, p);
        int is_inside = (e0 >= 0) & (e1 >= 0) & (e2 >= 0);

        // --- Apply Selected Reduction Strategy via Preprocessor ---
        #if defined(USE_GLOBAL_ATOMIC)
            if (is_inside) {
                atomicAdd(&out_pixel_counts[i], 1);
            }
        #elif defined(USE_WARP_REDUCE_TO_GLOBAL)
            unsigned int warp_mask = __activemask();
            int warp_sum = __popc(__ballot_sync(warp_mask, is_inside));
            if ((threadIdx.x % warpSize) == 0 && warp_sum > 0) {
                atomicAdd(&out_pixel_counts[i], warp_sum);
            }
        #elif defined(USE_SHMEM_ATOMIC_THEN_COPY)
            // unsigned int warp_mask = __activemask();
            // int warp_sum = __popc(__ballot_sync(warp_mask, is_inside));

            // if (thread_id % warpSize == 0) {
            if (is_inside) {
                atomicAdd(&shmem_counts[i], 1);
            }
        #endif
    }

    // --- Final Copy for the Shared Memory Strategy ---
    #if defined(USE_SHMEM_ATOMIC_THEN_COPY)
        __syncthreads();
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            out_pixel_counts[i] = shmem_counts[i];
        }
    #endif
}

/**
 * @brief Main Rasterizer Kernel ("Smart" Method) - Unchanged
 */
// __global__ void fine_raster_kernel(const Triangle* triangles, int* out_pixel_counts) {
//     for (int i = threadIdx.x; i < N; i += blockDim.x) {
//         Triangle t = triangles[i];
//         int min_x = 0;
//         int max_x = TILE_DIM - 1;
//         int min_y = 0;
//         int max_y = TILE_DIM - 1;

//         int pixel_count = 0;
//         for (int y = min_y; y <= max_y; ++y) {
//             for (int x = min_x; x <= max_x; ++x) {
//                 int2 p = {x, y};
//                 if (edge_function(t.v0, t.v1, p) >= 0 &&
//                     edge_function(t.v1, t.v2, p) >= 0 && 
//                     edge_function(t.v2, t.v0, p) >= 0) {
//                     pixel_count++;
//                 }
//             }
//         }
//         out_pixel_counts[i] = pixel_count;
//     }
// }
__device__ __forceinline__ int orient2d(const int2 &a,
                                        const int2 &b,
                                        const int2 &c)
{
    // (c - a) x (b - a)
    return (c.x - a.x) * (b.y - a.y) -
           (c.y - a.y) * (b.x - a.x);
}
/*


*/
constexpr int STEP = 4; // Size of the sub-tiles for hierarchical rasterization

#include <cuda_runtime.h>

// A work item for a quad that partially overlaps a triangle
struct PartialQuad {
    unsigned short tri_idx_in_block; // Triangle index within the block (0-255)
    unsigned char qx, qy;             // Quad's top-left coordinates
};

__global__ void fine_raster_kernel(const Triangle *__restrict__ triangles,
                                                int *__restrict__ out_pixel_counts)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    // 1. Load one triangle into registers (same as original)
    Triangle tri = triangles[tid];
    const int2 v0 = tri.v0;
    const int2 v1 = tri.v1;
    const int2 v2 = tri.v2;

    // 2. Pre-compute edge deltas for the incremental walk (same as original)
    const int A12 = v1.y - v2.y,  B12 = v2.x - v1.x;
    const int A20 = v2.y - v0.y,  B20 = v0.x - v2.x;
    const int A01 = v0.y - v1.y,  B01 = v1.x - v0.x;

    int pixel_count = 0;

    // 3. Hierarchical Raster Loop
    for (int y = 0; y < TILE_DIM; y += STEP)
    {
        for (int x = 0; x < TILE_DIM; x += STEP)
        {
            // 3.1. Evaluate edge functions at the quad's four corners
            int2 p00 = {x, y};
            int2 p10 = {x + STEP - 1, y};
            int2 p01 = {x, y + STEP - 1};
            int2 p11 = {x + STEP - 1, y + STEP - 1};

            int w0_p00 = orient2d(v1, v2, p00); int w1_p00 = orient2d(v2, v0, p00); int w2_p00 = orient2d(v0, v1, p00);
            int w0_p10 = orient2d(v1, v2, p10); int w1_p10 = orient2d(v2, v0, p10); int w2_p10 = orient2d(v0, v1, p10);
            int w0_p01 = orient2d(v1, v2, p01); int w1_p01 = orient2d(v2, v0, p01); int w2_p01 = orient2d(v0, v1, p01);
            int w0_p11 = orient2d(v1, v2, p11); int w1_p11 = orient2d(v2, v0, p11); int w2_p11 = orient2d(v0, v1, p11);

            // 3.2. Trivial Reject Check
            // If all 4 corners are "outside" any single edge, the whole quad is out.
            // A bitwise OR is negative iff all its components are negative.
            if (((w0_p00 | w0_p10 | w0_p01 | w0_p11) < 0) ||
                ((w1_p00 | w1_p10 | w1_p01 | w1_p11) < 0) ||
                ((w2_p00 | w2_p10 | w2_p01 | w2_p11) < 0))
            {
                continue; // Skip to next quad
            }

            // 3.3. Trivial Accept Check
            // If all 4 corners are "inside" all three edges, the whole quad is in.
            // A bitwise AND is non-negative if none of the components are negative.
            // Note: This is a conservative check. A more accurate check is (w0|w1|w2...)>=0 but
            // the AND is often faster as it can fail early. We will use the safer OR version.
            if (((w0_p00 | w0_p10 | w0_p01 | w0_p11) >= 0) &&
                ((w1_p00 | w1_p10 | w1_p01 | w1_p11) >= 0) &&
                ((w2_p00 | w2_p10 | w2_p01 | w2_p11) >= 0))
            {
                pixel_count += (STEP * STEP);
                continue; // Skip to next quad
            }

            // 3.4. Partial Overlap: Fallback to per-pixel testing inside the quad
            int w0_row = w0_p00;
            int w1_row = w1_p00;
            int w2_row = w2_p00;

            for (int iy = 0; iy < STEP; ++iy)
            {
                int w0 = w0_row;
                int w1 = w1_row;
                int w2 = w2_row;
                for (int ix = 0; ix < STEP; ++ix)
                {
                    if ((w0 | w1 | w2) >= 0)
                    {
                        ++pixel_count;
                    }
                    w0 += A12; w1 += A20; w2 += A01;
                }
                w0_row += B12; w1_row += B20; w2_row += B01;
            }
        }
    }

    out_pixel_counts[tid] = pixel_count;
}

/**
 * @brief A rasterizer using a single-kernel, two-pass approach with shared memory
 * to minimize warp divergence.
 *
 * @tparam STEP The dimension of the quad to test (e.g., 4 for a 4x4 quad).
 * @tparam BLOCK_DIM The number of threads in the CUDA block (e.g., 256).
 */
constexpr int BLOCK_DIM_NTHREADS = 256;
__global__ void fine_raster_smem_kernel(const Triangle *__restrict__ triangles,
                                        int *__restrict__ out_pixel_counts)
{
    // --- Shared Memory Declaration ---
    // Cache for the triangles this block is responsible for
    __shared__ Triangle triangle_cache[BLOCK_DIM_NTHREADS];
    // Work queue for quads that need detailed per-pixel processing
    __shared__ PartialQuad work_queue[BLOCK_DIM_NTHREADS * 8]; // Heuristic: allow for 8 partial quads/tri on avg
    // Counter for the work queue size
    __shared__ int queue_size;
    // Per-thread pixel counts for the block
    __shared__ int block_pixel_counts[BLOCK_DIM_NTHREADS];

    // --- Initialization ---
    const int ltid = threadIdx.x; // Local thread ID (0 to BLOCK_DIM-1)
    const int global_tid = blockIdx.x * BLOCK_DIM_NTHREADS + ltid;

    // One thread initializes the queue counter
    if (ltid == 0) {
        queue_size = 0;
    }
    // Each thread initializes its own count and loads one triangle into the shared cache
    if (global_tid < N) {
        block_pixel_counts[ltid] = 0;
        triangle_cache[ltid] = triangles[global_tid];
    }
    __syncthreads(); // Ensure all initialization and loading is complete

    if (global_tid >= N) return;

    // ====================================================================
    // PASS 1: CLASSIFY QUADS (Trivial Accept/Reject vs. Partial)
    // ====================================================================
    Triangle tri = triangle_cache[ltid];
    const int2 v0 = tri.v0, v1 = tri.v1, v2 = tri.v2;

    for (int y = 0; y < TILE_DIM; y += STEP) {
        for (int x = 0; x < TILE_DIM; x += STEP) {
            int2 p00 = {x, y}, p10 = {x + STEP - 1, y}, p01 = {x, y + STEP - 1}, p11 = {x + STEP - 1, y + STEP - 1};
            int w0_corners = orient2d(v1, v2, p00) | orient2d(v1, v2, p10) | orient2d(v1, v2, p01) | orient2d(v1, v2, p11);
            int w1_corners = orient2d(v2, v0, p00) | orient2d(v2, v0, p10) | orient2d(v2, v0, p01) | orient2d(v2, v0, p11);
            int w2_corners = orient2d(v0, v1, p00) | orient2d(v0, v1, p10) | orient2d(v0, v1, p01) | orient2d(v0, v1, p11);

            if (w0_corners < 0 || w1_corners < 0 || w2_corners < 0) continue; // Trivial Reject

            if (w0_corners >= 0 && w1_corners >= 0 && w2_corners >= 0) {
                block_pixel_counts[ltid] += (STEP * STEP); // Trivial Accept
            } else {
                // Partial Overlap: Add to the shared work queue
                int idx = atomicAdd(&queue_size, 1);
                // NOTE: In a production scenario, you must handle queue overflow!
                if (idx < BLOCK_DIM_NTHREADS * 8) {
                    work_queue[idx] = {(unsigned short)ltid, (unsigned char)x, (unsigned char)y};
                }
            }
        }
    }
    __syncthreads(); // Ensure Pass 1 is complete and queue is fully populated


    // ====================================================================
    // PASS 2: PROCESS PARTIAL QUADS (Highly Coherent)
    // ====================================================================
    const int A12 = v1.y - v2.y, B12 = v2.x - v1.x;
    const int A20 = v2.y - v0.y, B20 = v0.x - v2.x;
    const int A01 = v0.y - v1.y, B01 = v1.x - v0.x;

    // Threads in the block collectively process the shared work queue
    for (int i = ltid; i < queue_size; i += BLOCK_DIM_NTHREADS) {
        PartialQuad work = work_queue[i];
        Triangle partial_tri = triangle_cache[work.tri_idx_in_block];
        int2 p = {work.qx, work.qy};

        int w0_row = orient2d(partial_tri.v1, partial_tri.v2, p);
        int w1_row = orient2d(partial_tri.v2, partial_tri.v0, p);
        int w2_row = orient2d(partial_tri.v0, partial_tri.v1, p);
        
        int partial_pixels = 0;
        for (int iy = 0; iy < STEP; ++iy) {
            int w0 = w0_row, w1 = w1_row, w2 = w2_row;
            for (int ix = 0; ix < STEP; ++ix) {
                if ((w0 | w1 | w2) >= 0) partial_pixels++;
                w0 += A12; w1 += A20; w2 += A01;
            }
            w0_row += B12; w1_row += B20; w2_row += B01;
        }

        if (partial_pixels > 0) {
            atomicAdd(&block_pixel_counts[work.tri_idx_in_block], partial_pixels);
        }
    }
    __syncthreads(); // Ensure all threads are done processing partials


    // --- Final Write-back ---
    // Each thread writes its final count to global memory (coalesced)
    out_pixel_counts[global_tid] = block_pixel_counts[ltid];
}

// LucidRaster-inspired hierarchical dimensions for a 32x32 tile (or "bin")
// constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROW_HEIGHT = 8;
constexpr int BLOCK_DIM = 8;
constexpr int HALF_BLOCK_HEIGHT = 4;
constexpr int HALF_BLOCK_WIDTH = 8;

#include <cuda_runtime.h>

// Note: The 'Triangle', 'int2', and 'orient2d' definitions are assumed to be available
// as they were in the context of the original kernel.

__global__ void fine_raster_kernel_new(const Triangle *__restrict__ triangles,
                                   int *__restrict__ out_pixel_counts)
{
    // In a full implementation, this would be a thread in a 256-thread workgroup
    // processing one of many triangles assigned to a bin.
    // For this example, we maintain the 1-thread-per-triangle model.
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    // 1. Load one triangle into registers (same as original)
    Triangle tri = triangles[tid];
    const int2 v0 = tri.v0;
    const int2 v1 = tri.v1;
    const int2 v2 = tri.v2;

    // 2. Pre-compute edge deltas for the incremental walk (same as original)
    const int A12 = v1.y - v2.y,  B12 = v2.x - v1.x;
    const int A20 = v2.y - v0.y,  B20 = v0.x - v2.x;
    const int A01 = v0.y - v1.y,  B01 = v1.x - v0.x;

    int pixel_count = 0;

    // 3. Hierarchical Raster Loop (following LucidRaster decomposition)
    // The entire 32x32 tile is considered the "bin".

    // Loop over the 4 block-rows (32x8) in the bin
    for (int y_br = 0; y_br < TILE_DIM; y_br += BLOCK_ROW_HEIGHT)
    {
        // Loop over the 4 blocks (8x8) in the current block-row
        for (int x_b = 0; x_b < TILE_DIM; x_b += BLOCK_DIM)
        {
            // Loop over the 2 half-blocks (8x4) in the current block
            for (int y_hb_offset = 0; y_hb_offset < BLOCK_DIM; y_hb_offset += HALF_BLOCK_HEIGHT)
            {
                const int x_hb = x_b;
                const int y_hb = y_br + y_hb_offset;

                // --- Test the current 8x4 half-block ---

                // 3.1. Evaluate edge functions at the half-block's four corners
                int2 p00 = {x_hb, y_hb};
                int2 p10 = {x_hb + HALF_BLOCK_WIDTH - 1, y_hb};
                int2 p01 = {x_hb, y_hb + HALF_BLOCK_HEIGHT - 1};
                int2 p11 = {x_hb + HALF_BLOCK_WIDTH - 1, y_hb + HALF_BLOCK_HEIGHT - 1};

                int w0_p00 = orient2d(v1, v2, p00); int w1_p00 = orient2d(v2, v0, p00); int w2_p00 = orient2d(v0, v1, p00);
                int w0_p10 = orient2d(v1, v2, p10); int w1_p10 = orient2d(v2, v0, p10); int w2_p10 = orient2d(v0, v1, p10);
                int w0_p01 = orient2d(v1, v2, p01); int w1_p01 = orient2d(v2, v0, p01); int w2_p01 = orient2d(v0, v1, p01);
                int w0_p11 = orient2d(v1, v2, p11); int w1_p11 = orient2d(v2, v0, p11); int w2_p11 = orient2d(v0, v1, p11);

                // 3.2. Trivial Reject Check for the half-block
                // If all 4 corners are "outside" any single edge, the whole half-block is out.
                if (((w0_p00 | w0_p10 | w0_p01 | w0_p11) < 0) ||
                    ((w1_p00 | w1_p10 | w1_p01 | w1_p11) < 0) ||
                    ((w2_p00 | w2_p10 | w2_p01 | w2_p11) < 0))
                {
                    continue; // Skip to next half-block
                }

                // 3.3. Trivial Accept Check for the half-block
                // If all 4 corners are "inside" all three edges, the whole half-block is in.
                if (((w0_p00 | w0_p10 | w0_p01 | w0_p11) >= 0) &&
                    ((w1_p00 | w1_p10 | w1_p01 | w1_p11) >= 0) &&
                    ((w2_p00 | w2_p10 | w2_p01 | w2_p11) >= 0))
                {
                    pixel_count += (HALF_BLOCK_WIDTH * HALF_BLOCK_HEIGHT);
                    continue; // Skip to next half-block
                }

                // 3.4. Partial Overlap: Fallback to per-pixel testing inside the half-block
                // This corresponds to "half-block extraction" leading to pixel shading.
                int w0_row = w0_p00;
                int w1_row = w1_p00;
                int w2_row = w2_p00;

                for (int iy = 0; iy < HALF_BLOCK_HEIGHT; ++iy)
                {
                    int w0 = w0_row;
                    int w1 = w1_row;
                    int w2 = w2_row;
                    for (int ix = 0; ix < HALF_BLOCK_WIDTH; ++ix)
                    {
                        if ((w0 | w1 | w2) >= 0)
                        {
                            ++pixel_count;
                        }
                        w0 += A12; w1 += A20; w2 += A01; // Step one pixel to the right
                    }
                    w0_row += B12; w1_row += B20; w2_row += B01; // Step one pixel down
                }
            }
        }
    }

    out_pixel_counts[tid] = pixel_count;
}

// --- Host Code ---

void generate_random_triangles(std::vector<Triangle>& triangles, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, TILE_DIM);
    for (int i = 0; i < N; ++i) {
        Triangle& t = triangles[i];
        t.v0 = {dist(rng), dist(rng)};
        t.v1 = {dist(rng), dist(rng)};
        t.v2 = {dist(rng), dist(rng)};
        int cross_product = (t.v1.x - t.v0.x) * (t.v2.y - t.v0.y) - (t.v1.y - t.v0.y) * (t.v2.x - t.v0.x);
        if (cross_product < 0) {
            std::swap(t.v1, t.v2);
        }
    }
}

int main() {
    std::cout << "Starting Single-Tile Fine Rasterization Experiment" << std::endl;
    std::cout << "Tile Dimensions: " << TILE_DIM << "x" << TILE_DIM << ", Triangles: " << N << std::endl;

    std::string strategy_name;
    #if defined(USE_GLOBAL_ATOMIC)
        strategy_name = "GlobalAtomic";
    #elif defined(USE_WARP_REDUCE_TO_GLOBAL)
        strategy_name = "WarpReduceToGlobal";
    #elif defined(USE_SHMEM_ATOMIC_THEN_COPY)
        strategy_name = "ShmemAtomicThenCopy";
    #endif
    std::cout << "Reference Kernel Strategy: " << strategy_name << std::endl;
    
    // --- Setup and Initialization ---
    std::vector<Triangle> h_triangles(N);
    std::vector<int> h_ref_counts(N, 0);
    std::vector<int> h_fine_counts(N, 0);
    std::mt19937 rng(1337);

    // --- CUDA Memory Allocation ---
    Triangle* d_triangles;
    int* d_ref_counts;
    int* d_fine_counts;
    size_t triangles_size = N * sizeof(Triangle);
    size_t counts_size = N * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_triangles, triangles_size));
    CUDA_CHECK(cudaMalloc(&d_ref_counts, counts_size));
    CUDA_CHECK(cudaMalloc(&d_fine_counts, counts_size));

    // --- Kernel Launch Configuration ---
    dim3 ref_grid_dim(1);
    dim3 ref_block_dim(TILE_DIM * TILE_DIM);
    size_t ref_shmem_size = 0;
    #if defined(USE_SHMEM_ATOMIC_THEN_COPY)
        ref_shmem_size = counts_size;
    #endif
    
    dim3 fine_grid_dim(1);
    dim3 fine_block_dim(256);

    // --- Warmup Invocations ---
    std::cout << "Performing " << WARMUP_RUNS << " warmup runs..." << std::endl;
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        generate_random_triangles(h_triangles, rng);
        CUDA_CHECK(cudaMemcpy(d_triangles, h_triangles.data(), triangles_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_ref_counts, 0, counts_size));
        CUDA_CHECK(cudaMemset(d_fine_counts, 0, counts_size));
        
        reference_raster_kernel<<<ref_grid_dim, ref_block_dim, ref_shmem_size>>>(d_triangles, d_ref_counts);
        fine_raster_kernel<<<fine_grid_dim, fine_block_dim>>>(d_triangles, d_fine_counts);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // --- Timed Main Run ---
    std::cout << "Executing timed run..." << std::endl;
    generate_random_triangles(h_triangles, rng);
    CUDA_CHECK(cudaMemcpy(d_triangles, h_triangles.data(), triangles_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_ref_counts, 0, counts_size));
    CUDA_CHECK(cudaMemset(d_fine_counts, 0, counts_size));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms_ref = 0, ms_fine = 0;

    // Run Reference Kernel
    CUDA_CHECK(cudaEventRecord(start));
    reference_raster_kernel<<<ref_grid_dim, ref_block_dim, ref_shmem_size>>>(d_triangles, d_ref_counts);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_ref, start, stop));
    
    // Run Fine Rasterizer Kernel
    CUDA_CHECK(cudaEventRecord(start));
    fine_raster_kernel<<<fine_grid_dim, fine_block_dim>>>(d_triangles, d_fine_counts);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_fine, start, stop));

    // --- Verification and Results ---
    CUDA_CHECK(cudaMemcpy(h_ref_counts.data(), d_ref_counts, counts_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fine_counts.data(), d_fine_counts, counts_size, cudaMemcpyDeviceToHost));
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (h_ref_counts[i] != h_fine_counts[i]) mismatches++;
    }

    std::cout << "\n--- Performance Results ---" << std::endl;
    printf("Reference Kernel ('Dumb' - %s): %.6f ms\n", strategy_name.c_str(), ms_ref);
    printf("Fine Kernel ('Smart'):                 %.6f ms\n", ms_fine);
    if (ms_fine > 0) printf("Speedup (Smart vs Dumb): %.2fx\n", ms_ref / ms_fine);

    std::cout << "\n--- Correctness Check ---" << std::endl;
    if (mismatches == 0) std::cout << "SUCCESS: The outputs of both kernels match perfectly." << std::endl;
    else std::cout << "FAILURE: Found " << mismatches << " mismatches out of " << N << " triangles." << std::endl;
    
    // --- Cleanup ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_ref_counts));
    CUDA_CHECK(cudaFree(d_fine_counts));

    return 0;
}