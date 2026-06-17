# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
#
# Runtime test suite. Linked against the gpu_runtime target built by the
# parent CMakeLists. Each test is a small driver program that exits 0 on
# success; tests gracefully skip when no GPU backend is available at runtime
# rather than failing the suite, so the same binaries are usable on CI hosts
# without a GPU.

if(NOT TARGET gpu_runtime)
    message(FATAL_ERROR "runtime/tests must be built via the runtime CMakeLists.txt "
                        "(missing target gpu_runtime)")
endif()

set(GPU_RUNTIME_TESTS
    test_device_selection
    test_events
    test_graphs
    test_kernel_execution
    test_program_cache
    test_memory_resources
    test_stream_pool
    test_device_buffer
)

foreach(test_name IN LISTS GPU_RUNTIME_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE gpu_runtime)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Convenience: build all tests in one shot.
add_custom_target(gpu_runtime_tests
    DEPENDS ${GPU_RUNTIME_TESTS}
)
