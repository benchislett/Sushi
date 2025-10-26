#!/bin/bash

DEBUG=0

BUILD_TYPE=Release
if [ $DEBUG -eq 1 ]; then
    BUILD_TYPE=Debug
fi

cmake --build build_manual --config $BUILD_TYPE
