add_library(sushi_common INTERFACE)

target_include_directories(sushi_common
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
