# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.26)
project(navatala_gpu_python LANGUAGES CXX)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

set(NAVATALA_FFI_INCLUDE_DIR "" CACHE PATH "Directory containing navatala_ffi.h")
set(NAVATALA_FFI_LIBRARY "" CACHE FILEPATH "Path to the public navatala GPU FFI library")
if (NOT NAVATALA_FFI_INCLUDE_DIR)
    message(FATAL_ERROR "NAVATALA_FFI_INCLUDE_DIR must point to the public gpu_runtime/include directory")
endif()
if (NOT NAVATALA_FFI_LIBRARY)
    message(FATAL_ERROR "NAVATALA_FFI_LIBRARY must point to the public gpu_runtime shared library")
endif()
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

add_subdirectory(bindings)
