#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// Helper macros for checking CUDA errors
#define CUDA_CHECK(err)                                                                  \
    if (err != cudaSuccess)                                                              \
    {                                                                                    \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    }

#define CUDA_CHECK_LAST_ERROR()                                                              \
    {                                                                                        \
        cudaError_t err = cudaGetLastError();                                                \
        if (err != cudaSuccess)                                                              \
        {                                                                                    \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        }                                                                                    \
    }

#define CUDA_SYNC_CHECK()                                                                                                     \
    {                                                                                                                         \
        cudaError_t err = cudaDeviceSynchronize();                                                                            \
        if (err != cudaSuccess)                                                                                               \
        {                                                                                                                     \
            throw std::runtime_error(std::string("CUDA Error identified after synchronization: ") + cudaGetErrorString(err)); \
        }                                                                                                                     \
    }
