/*
This is a benchmark and test script for compositing implementations using CUDA.

It sets up two images (background and foreground) and measures the performance of different compositing methods.
- Background will have RGB UINT8 color values [0,255].
- Foreground will have RGBA UINT8 color values [0,255], with non-trivial alpha (never zero, never 255).

The script tests different compositing methods using premultiplied alpha with various implementations.
Each runs on a 1024x1024 image for a specified number of iterations to gather timing data.

For testing, the script compares the output of each method against a reference implementation to ensure correctness.
If any discrepancies of more than 1 unit in any color channel are found, an error is raised.
*/

#include <sushi_cuda/cuda_utils.cuh>

#include <iostream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>

const int WIDTH = 1024;
const int HEIGHT = 1024;
const int ITERATIONS = 100;
const unsigned int SEED = 42;

bool check_images_equal(
    const ImageRGB& img1, const ImageRGB& img2
) {
    if (img1.width != img2.width || img1.height != img2.height) {
        std::cerr << "Error: Image dimensions mismatch.\n";
        return false;
    }
    int total_pixels = img1.width * img1.height;
    int errors_found = 0;
    for (int i = 0; i < total_pixels; ++i) {
        for(int c = 0; c < 3; ++c) { // R, G, B
            int idx = i * 3 + c;
            int diff = std::abs(static_cast<int>(img1.data[idx]) - static_cast<int>(img2.data[idx]));
            if (diff > 1) { // Allowable tolerance of 1
                if (errors_found < 10) { // Print first 10 errors
                    std::cerr << "Verification FAILED at pixel " << i << ", channel " << c
                              << ". Ref: " << static_cast<int>(img1.data[idx])
                              << ", Test: " << static_cast<int>(img2.data[idx])
                              << ", Diff: " << diff << "\n";
                }
                errors_found++;
            }
        }
    }

    if(errors_found > 0) {
        std::cerr << "Total errors found: " << errors_found << "\n";
        return false;
    }
    return true;
}

ImageRGB composit_cpu_reference(
    const ImageRGB& background,
    const ImageRGBA& foreground
) {
    ImageRGB output;
    output.width = background.width;
    output.height = background.height;
    int total_pixels = output.width * output.height;
    output.data = new uint8_t[total_pixels * 3];

    for (int i = 0; i < total_pixels; ++i) {
        float alpha = foreground.data[i * 4 + 3] / 255.0f; // Normalize alpha
        float inv_alpha = 1.0f - alpha;
        for (int c = 0; c < 3; ++c) { // For R, G, B channels
            float fg_val = foreground.data[i * 4 + c]; // Foreground color is already premultiplied
            float bg_val = background.data[i * 3 + c] * inv_alpha;
            // Add 0.5f for correct rounding before truncation
            output.data[i * 3 + c] = static_cast<uint8_t>(std::min(255.0f, fg_val + bg_val + 0.5f));
        }
    }
    return output;
}

__global__ void composit_cuda_kernel(
    uchar4* output,
    const uchar4* background,
    const uchar4* foreground,
    int total_pixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_pixels) return;

    uchar4 fg_pix = foreground[i];

    // Background and output alpha channels are ignored
    uchar4 bg_pix = background[i];

    uint8_t inv_a = 255 - fg_pix.w;

    uint8_t out_r = fg_pix.x + ((bg_pix.x * inv_a + 128) >> 8);
    uint8_t out_g = fg_pix.y + ((bg_pix.y * inv_a + 128) >> 8);
    uint8_t out_b = fg_pix.z + ((bg_pix.z * inv_a + 128) >> 8);

    // Output alpha is irrelevant, just set to 255
    output[i] = make_uchar4(out_r, out_g, out_b, 255);
}

void run_composit_cuda(
    ImageRGBA& d_output,
    const ImageRGBA& d_background,
    const ImageRGBA& d_foreground)
{
    int total_pixels = d_background.width * d_background.height;
    int threads_per_block = 256;
    int blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

    composit_cuda_kernel<<<blocks, threads_per_block>>>(
        (uchar4*)d_output.data,
        (uchar4*)d_background.data,
        (uchar4*)d_foreground.data,
        total_pixels
    );
    CUDA_CHECK_LAST_ERROR();
}

int size_bytes(const ImageRGB& img) {
    return img.width * img.height * 3 * sizeof(uint8_t);
}

int size_bytes(const ImageRGBA& img) {
    return img.width * img.height * 4 * sizeof(uint8_t);
}

void allocate_host_memory(ImageRGB& img, int w, int h) {
    img.width = w;
    img.height = h;
    img.data = new uint8_t[size_bytes(img)];
}

void allocate_host_memory(ImageRGBA& img, int w, int h) {
    img.width = w;
    img.height = h;
    img.data = new uint8_t[size_bytes(img)];
}

void free_host_memory(ImageRGB& img) {
    delete[] img.data;
    img.data = nullptr;
}

void free_host_memory(ImageRGBA& img) {
    delete[] img.data;
    img.data = nullptr;
}

void initialize_images(ImageRGB& background, ImageRGBA& foreground) {
    std::mt19937 gen(SEED);
    std::uniform_int_distribution<int> dist_rgb(0, 255);
    std::uniform_int_distribution<int> dist_alpha(1, 254); // Non-trivial alpha [1, 254]

    int total_pixels = background.width * background.height;

    // Initialize background (RGB)
    for (int i = 0; i < total_pixels; ++i) {
        background.data[i * 3 + 0] = static_cast<uint8_t>(dist_rgb(gen));
        background.data[i * 3 + 1] = static_cast<uint8_t>(dist_rgb(gen));
        background.data[i * 3 + 2] = static_cast<uint8_t>(dist_rgb(gen));
    }

    // Initialize foreground (RGBA, premultiplied)
    for (int i = 0; i < total_pixels; ++i) {
        // 1. Generate "true" (non-premultiplied) color and alpha
        uint8_t r = static_cast<uint8_t>(dist_rgb(gen));
        uint8_t g = static_cast<uint8_t>(dist_rgb(gen));
        uint8_t b = static_cast<uint8_t>(dist_rgb(gen));
        uint8_t a = static_cast<uint8_t>(dist_alpha(gen));
        float alpha_f = a / 255.0f;

        // 2. Premultiply: C_pre = C_true * alpha
        // Add 0.5f for correct rounding when casting from float to int
        foreground.data[i * 4 + 0] = static_cast<uint8_t>(r * alpha_f + 0.5f); // R
        foreground.data[i * 4 + 1] = static_cast<uint8_t>(g * alpha_f + 0.5f); // G
        foreground.data[i * 4 + 2] = static_cast<uint8_t>(b * alpha_f + 0.5f); // B
        foreground.data[i * 4 + 3] = a;                                        // A
    }
}


void allocate_device_memory(ImageRGBA& img, int w, int h) {
    img.width = w;
    img.height = h;
    CUDA_CHECK(cudaMalloc(&img.data, size_bytes(img)));
}

void free_device_memory(ImageRGBA& img) {
    if (img.data) cudaFree(img.data);
    img.data = nullptr;
}

void upload_to_device(ImageRGBA& d_img, const ImageRGB& h_img) {
    int total_pixels = h_img.width * h_img.height;
    // Create a temporary host buffer for RGBA conversion
    uint8_t* new_data = new uint8_t[size_bytes(d_img)];
    for (int i = 0; i < total_pixels; ++i) {
        new_data[i * 4 + 0] = h_img.data[i * 3 + 0]; // R
        new_data[i * 4 + 1] = h_img.data[i * 3 + 1]; // G
        new_data[i * 4 + 2] = h_img.data[i * 3 + 2]; // B
        new_data[i * 4 + 3] = 255;                   // A (unused, set to 255)
    }
    // Upload the 4-channel data
    CUDA_CHECK(cudaMemcpy(d_img.data, new_data, size_bytes(d_img), cudaMemcpyHostToDevice));
    CUDA_SYNC_CHECK();
    delete[] new_data;
}

// This overload for RGBA-to-RGBA copies is unchanged and correct.
void upload_to_device(ImageRGBA& d_img, const ImageRGBA& h_img) {
    CUDA_CHECK(cudaMemcpy(d_img.data, h_img.data, size_bytes(h_img), cudaMemcpyHostToDevice));
}

void download_from_device(ImageRGB& h_img, const ImageRGBA& d_img) {
    // Create a temporary host buffer to receive the 4-channel RGBA data
    uint8_t* new_data = new uint8_t[size_bytes(d_img)];
    CUDA_CHECK(cudaMemcpy(new_data, d_img.data, size_bytes(d_img), cudaMemcpyDeviceToHost));
    CUDA_SYNC_CHECK();

    // Copy R, G, B channels to the 3-channel host destination
    int total_pixels = d_img.width * d_img.height;
    for (int i = 0; i < total_pixels; ++i) {
        h_img.data[i * 3 + 0] = new_data[i * 4 + 0]; // R
        h_img.data[i * 3 + 1] = new_data[i * 4 + 1]; // G
        h_img.data[i * 3 + 2] = new_data[i * 4 + 2]; // B
    }
    delete[] new_data;
}

class BenchmarkTimer {
public:
    enum TimerType { CPU, GPU };

    BenchmarkTimer(TimerType type = GPU) : type_(type) {}

    ~BenchmarkTimer() {
        if (type_ == GPU) {
            if (start_event_) cudaEventDestroy(start_event_);
            if (stop_event_) cudaEventDestroy(stop_event_);
        }
    }

    void create() {
        if (type_ == GPU) {
            CUDA_CHECK(cudaEventCreate(&start_event_));
            CUDA_CHECK(cudaEventCreate(&stop_event_));
        }
    }

    void start() {
        if (type_ == CPU) {
            start_time_ = std::chrono::high_resolution_clock::now();
        } else {
            CUDA_CHECK(cudaEventRecord(start_event_, 0));
        }
    }

    void stop() {
        if (type_ == CPU) {
            end_time_ = std::chrono::high_resolution_clock::now();
        } else {
            CUDA_CHECK(cudaEventRecord(stop_event_, 0));
        }
    }

    double elapsed_ms() {
        if (type_ == CPU) {
            return std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
        } else {
            CUDA_CHECK(cudaEventSynchronize(stop_event_));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_event_, stop_event_));
            return static_cast<double>(ms);
        }
    }

private:
    TimerType type_;
    // CPU
    std::chrono::high_resolution_clock::time_point start_time_, end_time_;
    // GPU
    cudaEvent_t start_event_ = nullptr, stop_event_ = nullptr;
};

void log_results(const std::string& name, double total_ms, int iterations, long long total_pixels) {
    double avg_ms = total_ms / iterations;
    double total_pixels_processed = static_cast<double>(total_pixels) * iterations;
    double pixels_per_sec = total_pixels_processed / (total_ms / 1000.0);
    double giga_pixels_per_sec = pixels_per_sec / 1e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  " << std::setw(20) << std::left << name + ":"
              << std::setw(10) << std::right << avg_ms << " ms/iter"
              << std::setw(12) << std::right << giga_pixels_per_sec << " Gpix/s"
              << "  (Total: " << total_ms << " ms for " << iterations << " iter)" << std::endl;
}

int main() {
    try {
        const long long total_pixels = static_cast<long long>(WIDTH) * HEIGHT;
        std::cout << "Alpha Blending Benchmark\n";
        std::cout << "Image Size: " << WIDTH << "x" << HEIGHT << " (" << total_pixels << " pixels)\n";
        std::cout << "Iterations: " << ITERATIONS << "\n";
        std::cout << "------------------------------------------------------------\n";

        // 1. Setup
        std::cout << "Setting up...\n";
        // Host Memory (Background/Output are 3-channel RGB)
        ImageRGB h_background, h_output_ref, h_output_test;
        ImageRGBA h_foreground;
        allocate_host_memory(h_background, WIDTH, HEIGHT);
        allocate_host_memory(h_foreground, WIDTH, HEIGHT);
        allocate_host_memory(h_output_test, WIDTH, HEIGHT);
        // h_output_ref is allocated by the reference function

        // Device Memory (All buffers are 4-channel RGBA for performance)
        ImageRGBA d_background, d_output;
        ImageRGBA d_foreground;

        allocate_device_memory(d_background, WIDTH, HEIGHT);
        allocate_device_memory(d_foreground, WIDTH, HEIGHT);
        allocate_device_memory(d_output, WIDTH, HEIGHT);

        // Initialization & Upload
        std::cout << "Initializing random images (premultiplied)...\n";
        initialize_images(h_background, h_foreground);

        std::cout << "Uploading to GPU...\n";
        upload_to_device(d_background, h_background);
        upload_to_device(d_foreground, h_foreground);

        // 2. Run CPU Reference (Used for both timing and verification)
        std::cout << "\n--- Running & Benchmarking CPU Reference ---\n";
        BenchmarkTimer cpu_timer(BenchmarkTimer::CPU);
        cpu_timer.start();
        for (int i = 0; i < ITERATIONS; ++i) {
            if (h_output_ref.data) free_host_memory(h_output_ref); // Free previous run's data
            h_output_ref = composit_cpu_reference(h_background, h_foreground);
        }
        cpu_timer.stop();
        log_results("CPU Reference", cpu_timer.elapsed_ms(), ITERATIONS, total_pixels);


        // 3. Run CUDA Implementation
        std::cout << "\n--- Benchmarking CUDA Implementation ---\n";
        BenchmarkTimer gpu_timer(BenchmarkTimer::GPU);
        gpu_timer.create();

        // Warm-up run
        run_composit_cuda(d_output, d_background, d_foreground);
        CUDA_SYNC_CHECK();

        // Timed runs
        gpu_timer.start();
        for (int i = 0; i < ITERATIONS; ++i) {
            run_composit_cuda(d_output, d_background, d_foreground);
        }
        gpu_timer.stop();
        log_results("CUDA Implementation", gpu_timer.elapsed_ms(), ITERATIONS, total_pixels);

        // 4. Verify CUDA Implementation
        std::cout << "\n--- Verifying CUDA Implementation ---\n";
        download_from_device(h_output_test, d_output);
        if (check_images_equal(h_output_ref, h_output_test)) {
            std::cout << "Verification PASSED.\n";
        } else {
            std::cerr << "Verification FAILED.\n";
        }

        // 5. Teardown
        std::cout << "------------------------------------------------------------\n";
        std::cout << "Cleaning up...\n";
        free_host_memory(h_background);
        free_host_memory(h_foreground);
        free_host_memory(h_output_ref);
        free_host_memory(h_output_test);

        free_device_memory(d_background);
        free_device_memory(d_foreground);
        free_device_memory(d_output);

        CUDA_CHECK(cudaDeviceReset());
        std::cout << "Done.\n";

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
