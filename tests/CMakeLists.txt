# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd

find_package(Python3 COMPONENTS Interpreter QUIET)

if(Python3_Interpreter_FOUND)
    add_test(
        NAME manifest_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/manifest_conformance.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/..")
    add_test(
        NAME rocm_benchmark_json_fixture_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/rocm_vendor_benchmark.sample.json")
    add_test(
        NAME rocm_benchmark_mi300x_reference_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/rocm_vendor_benchmark_mi300x_20260619.json"
            "--require-full"
            "--require-hipsparselt")
    add_test(
        NAME rocm_benchmark_report_fixture_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/rocm_vendor_benchmark.sample.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark.sample.md")
    add_test(
        NAME rocm_benchmark_mi300x_reference_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/rocm_vendor_benchmark_mi300x_20260619.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_mi300x_20260619.md")
else()
    message(WARNING "Python3 interpreter not found; manifest_conformance test skipped")
endif()

if(TARGET gpu_runtime)
    add_executable(ffi_gemm_f16_f32_fallback_test
        ffi_gemm_f16_f32_fallback_test.cpp)
    target_link_libraries(ffi_gemm_f16_f32_fallback_test PRIVATE gpu_runtime)
    target_include_directories(ffi_gemm_f16_f32_fallback_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include")
    add_test(
        NAME ffi_gemm_f16_f32_fallback
        COMMAND ffi_gemm_f16_f32_fallback_test)
endif()
