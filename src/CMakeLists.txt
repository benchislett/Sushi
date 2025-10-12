cmake_minimum_required(VERSION 3.20)

project(sushi LANGUAGES CXX CUDA)

set(CMAKE_CXX_COMPILER g++-12)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find Python and nanobind
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

# Build options
option(SUSHI_BUILD_CUDA "Enable CUDA support" ON)

add_subdirectory(sushi_common)

add_subdirectory(sushi_cpp)

if(SUSHI_BUILD_CUDA)
  add_subdirectory(sushi_cuda)
endif()

add_subdirectory(sushi_bindings)







# # We are now ready to compile the actual extension module
# nanobind_add_module(
#   # Name of the extension
#   sushi_core

#   # Optimize for performance instead of size
#   NOMINSIZE

#   # Target the stable ABI for Python 3.12+, which reduces
#   # the number of binary wheels that must be built. This
#   # does nothing on older Python versions
#   STABLE_ABI

#   # Source code goes here
#   sushi_core_main.cpp
# )

# # Ensure proper include directories
# target_include_directories(sushi_core PRIVATE ${nanobind_INCLUDE_DIRS})

# # Install directive for scikit-build-core
# install(TARGETS sushi_core LIBRARY DESTINATION sushi)

# # set max optimizations for nvcc
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# add_executable(test_sushi_core test_cuda.cu)
# set_property(TARGET test_sushi_core PROPERTY CUDA_ARCHITECTURES native)
