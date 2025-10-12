add_library(sushi_cuda STATIC
  src/cuda_backend.cu
)

target_link_libraries(sushi_cuda PUBLIC
  sushi_common
)

target_include_directories(sushi_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    $<TARGET_PROPERTY:sushi_cpp,INTERFACE_INCLUDE_DIRECTORIES>
)
