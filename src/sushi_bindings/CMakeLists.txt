nanobind_add_module(sushi_core
  NOMINSIZE
  LTO
  bindings.cpp
)

target_link_libraries(sushi_core PRIVATE sushi_cpp)

if(SUSHI_BUILD_CUDA)
  target_link_libraries(sushi_core PRIVATE sushi_cuda)
  # Define a macro so we can conditionally compile CUDA-specific code
  target_compile_definitions(sushi_core PRIVATE WITH_CUDA)
endif()

target_include_directories(sushi_core PRIVATE
  $<TARGET_PROPERTY:sushi_core,INTERFACE_INCLUDE_DIRECTORIES>
)
if(SUSHI_BUILD_CUDA)
  target_include_directories(sushi_core PRIVATE
    $<TARGET_PROPERTY:sushi_cuda,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif()

install(TARGETS sushi_core LIBRARY DESTINATION sushi)
