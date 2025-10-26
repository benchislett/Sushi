#!/bin/bash

DEBUG=0

source .venv/bin/activate
NANOBIND_DIR=$(python3 -c "import nanobind; print(nanobind.cmake_dir())")
PYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)")
BUILD_TYPE=Release
if [ $DEBUG -eq 1 ]; then
    BUILD_TYPE=Debug
fi

cmake -B build_manual -S src \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_PREFIX_PATH=$NANOBIND_DIR \
    -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE

cmake --build build_manual --config $BUILD_TYPE
