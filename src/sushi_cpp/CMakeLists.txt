add_library(sushi_cpp STATIC
  src/rasterizer.cpp
)

target_link_libraries(sushi_cpp PUBLIC sushi_common)

target_include_directories(sushi_cpp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
