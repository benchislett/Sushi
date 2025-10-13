enable_language(CUDA)

nanobind_add_module(sushi_cuda STATIC NOMINSIZE LTO
  src/cuda_backend.cu
  include/sushi_cuda/cuda_backend.cuh
  include/sushi_cuda/cuda_utils.cuh
)

target_link_libraries(sushi_cuda PUBLIC
  sushi_common
  CUDA::cudart
)

set_property(TARGET sushi_cuda PROPERTY CUDA_ARCHITECTURES native)

target_include_directories(sushi_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    $<TARGET_PROPERTY:sushi_cpp,INTERFACE_INCLUDE_DIRECTORIES>
)
