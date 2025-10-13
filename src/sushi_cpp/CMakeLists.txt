nanobind_add_module(sushi_cpp STATIC NOMINSIZE LTO
  src/rasterizer.cpp
  include/sushi/rasterizer.h
)

target_link_libraries(sushi_cpp PUBLIC sushi_common)

target_include_directories(sushi_cpp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
