# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
#
# Build-time examples. Each example is a small, self-contained driver
# program that exercises one piece of the runtime + kernel corpus. Examples
# gracefully skip (exit 0) when the necessary backend isn't available so
# they're safe to register as CTest cases on CI hosts without a GPU.

if(NOT TARGET gpu_runtime)
    message(FATAL_ERROR
        "examples/ must be built alongside the runtime — set "
        "NAVATALA_GPU_BUILD_RUNTIME=ON.")
endif()

add_executable(axpy_example axpy_example.cpp)
target_link_libraries(axpy_example PRIVATE gpu_runtime)

# Register with CTest only when tests are enabled — keeps `ctest` from
# discovering an example case when the consumer asked for tests off.
# The example exits 0 when no real backend context is available, and it
# exercises the stub FFI in the default no-GPU build.
if(BUILD_TESTING OR GPU_RUNTIME_BUILD_TESTS)
    add_test(NAME axpy_example COMMAND axpy_example)
endif()
