# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
#
# Optional benchmark drivers. These targets are intentionally opt-in because
# they require vendor SDKs and real GPU hardware.

if(NOT NAVATALA_GPU_USE_HIP)
    message(STATUS "[Navatala GPU] ROCm benchmarks requested but HIP backend is disabled")
    return()
endif()

find_package(hip CONFIG QUIET HINTS /opt/rocm PATHS /opt/rocm)
if(NOT hip_FOUND)
    message(STATUS "[Navatala GPU] ROCm benchmarks requested but HIP CMake package was not found")
    return()
endif()

enable_language(HIP)

set(NAVATALA_GPU_ROCM_VERSION 0 CACHE STRING
    "ROCm version encoded as MAJOR*10000+MINOR*100+PATCH for generated HIP guards")
option(NAVATALA_GPU_BUILD_MFMA_BENCHMARK
    "Build opt-in HIP MFMA benchmark rows. Requires ROCm 6.2+ and gfx942-class hardware to run correctly."
    OFF)

function(_navatala_encode_rocm_version out_var raw_version)
    string(STRIP "${raw_version}" _navatala_rocm_raw)
    string(REGEX MATCH "([0-9]+)\\.([0-9]+)(\\.([0-9]+))?"
        _navatala_rocm_version_match "${_navatala_rocm_raw}")
    if(_navatala_rocm_version_match)
        set(_navatala_rocm_major "${CMAKE_MATCH_1}")
        set(_navatala_rocm_minor "${CMAKE_MATCH_2}")
        if(CMAKE_MATCH_4)
            set(_navatala_rocm_patch "${CMAKE_MATCH_4}")
        else()
            set(_navatala_rocm_patch "0")
        endif()
        math(EXPR _navatala_rocm_encoded
            "${_navatala_rocm_major} * 10000 + ${_navatala_rocm_minor} * 100 + ${_navatala_rocm_patch}")
        set(${out_var} "${_navatala_rocm_encoded}" PARENT_SCOPE)
    endif()
endfunction()

if(NAVATALA_GPU_ROCM_VERSION EQUAL 0)
    set(_navatala_rocm_prefixes ${CMAKE_PREFIX_PATH} /opt/rocm /opt/rocm-7.2.4 /opt/rocm-7.1.0 /opt/rocm-7.0.0 /opt/rocm-6.4.0 /opt/rocm-6.3.0 /opt/rocm-6.2.0)
    foreach(_navatala_rocm_prefix IN LISTS _navatala_rocm_prefixes)
        if(EXISTS "${_navatala_rocm_prefix}/.info/version-rocm")
            file(READ "${_navatala_rocm_prefix}/.info/version-rocm" _navatala_rocm_version_text)
            _navatala_encode_rocm_version(_navatala_rocm_detected "${_navatala_rocm_version_text}")
            if(_navatala_rocm_detected)
                set(NAVATALA_GPU_ROCM_VERSION "${_navatala_rocm_detected}" CACHE STRING
                    "ROCm version encoded as MAJOR*10000+MINOR*100+PATCH for generated HIP guards" FORCE)
                break()
            endif()
        endif()
    endforeach()
endif()

if(NAVATALA_GPU_ROCM_VERSION EQUAL 0 AND DEFINED hip_VERSION)
    # Last-resort fallback: some HIP packages expose package versions that are
    # not exactly the ROCm release (for example 7.2.53211). This is still good
    # enough to reject very old toolchains, but the .info/version-rocm path
    # above is preferred whenever available.
    _navatala_encode_rocm_version(_navatala_rocm_from_hip "${hip_VERSION}")
    if(_navatala_rocm_from_hip)
        set(NAVATALA_GPU_ROCM_VERSION "${_navatala_rocm_from_hip}" CACHE STRING
            "ROCm version encoded as MAJOR*10000+MINOR*100+PATCH for generated HIP guards" FORCE)
    endif()
endif()
if(NAVATALA_GPU_ROCM_VERSION GREATER 0)
    message(STATUS "[Navatala GPU] ROCm version guard: ${NAVATALA_GPU_ROCM_VERSION}")
else()
    message(STATUS "[Navatala GPU] ROCm version guard: not detected; generated HIP minimum-version guards are inactive")
endif()

set(NAVATALA_ROCM_BENCHMARK_KERNELS
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/sparse/axpy_f32.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/graph/spmv_weighted_f32.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/graph/spmv_weighted_subgroup_f32.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/transformer/tiled_gemm_f32.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/transformer/tiled_gemm_f16_f32_out.hip")

if(NAVATALA_GPU_BUILD_MFMA_BENCHMARK)
    list(APPEND NAVATALA_ROCM_BENCHMARK_KERNELS
        "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/transformer/tiled_gemm_f16_mfma_k_loop.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_direct.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_shared.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_shared_early_barrier.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_shared_padded.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_pipelined.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/transformer/tiled_gemm_f16_mfma_cta64_shared_edge.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/transformer/tiled_gemm_f16_mfma_cta128.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/hip/transformer/tiled_gemm_f16_mfma_cta128_edge.hip")
endif()

foreach(kernel_path IN LISTS NAVATALA_ROCM_BENCHMARK_KERNELS)
    if(NOT EXISTS "${kernel_path}")
        message(FATAL_ERROR
            "[Navatala GPU] ROCm benchmark kernel missing: ${kernel_path}. "
            "Regenerate the public tree with HIP kernels before enabling "
            "NAVATALA_GPU_BUILD_BENCHMARKS.")
    endif()
endforeach()

find_package(rocblas CONFIG QUIET HINTS /opt/rocm PATHS /opt/rocm)
find_package(rocsparse CONFIG QUIET HINTS /opt/rocm PATHS /opt/rocm)

set(NAVATALA_ROCM_VENDOR_LIBS "")
set(NAVATALA_ROCM_VENDOR_INCLUDES "")

if(TARGET roc::rocblas)
    list(APPEND NAVATALA_ROCM_VENDOR_LIBS roc::rocblas)
else()
    find_path(ROCBLAS_INCLUDE_DIR
        NAMES rocblas/rocblas.h rocblas.h
        HINTS /opt/rocm/include)
    find_library(ROCBLAS_LIBRARY NAMES rocblas HINTS /opt/rocm/lib /opt/rocm/lib64)
    if(ROCBLAS_LIBRARY AND ROCBLAS_INCLUDE_DIR)
        list(APPEND NAVATALA_ROCM_VENDOR_LIBS "${ROCBLAS_LIBRARY}")
        list(APPEND NAVATALA_ROCM_VENDOR_INCLUDES "${ROCBLAS_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR
            "[Navatala GPU] rocBLAS is required for rocm_vendor_benchmark")
    endif()
endif()

if(TARGET roc::rocsparse)
    list(APPEND NAVATALA_ROCM_VENDOR_LIBS roc::rocsparse)
else()
    find_path(ROCSPARSE_INCLUDE_DIR
        NAMES rocsparse/rocsparse.h rocsparse.h
        HINTS /opt/rocm/include)
    find_library(ROCSPARSE_LIBRARY NAMES rocsparse HINTS /opt/rocm/lib /opt/rocm/lib64)
    if(ROCSPARSE_LIBRARY AND ROCSPARSE_INCLUDE_DIR)
        list(APPEND NAVATALA_ROCM_VENDOR_LIBS "${ROCSPARSE_LIBRARY}")
        list(APPEND NAVATALA_ROCM_VENDOR_INCLUDES "${ROCSPARSE_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR
            "[Navatala GPU] rocSPARSE is required for rocm_vendor_benchmark")
    endif()
endif()

find_path(HIPSPARSELT_INCLUDE_DIR
    NAMES hipsparselt/hipsparselt.h hipsparseLt/hipsparseLt.h hipsparseLt.h
    HINTS /opt/rocm/include)
find_library(HIPSPARSELT_LIBRARY
    NAMES hipsparseLt hipsparselt
    HINTS /opt/rocm/lib /opt/rocm/lib64)

add_executable(rocm_vendor_benchmark rocm_vendor_benchmark.hip)
target_compile_features(rocm_vendor_benchmark PRIVATE cxx_std_17)
target_include_directories(rocm_vendor_benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include"
    ${NAVATALA_ROCM_VENDOR_INCLUDES})
target_link_libraries(rocm_vendor_benchmark PRIVATE
    hip::host
    ${NAVATALA_ROCM_VENDOR_LIBS})
if(NAVATALA_GPU_ROCM_VERSION GREATER 0)
    target_compile_definitions(rocm_vendor_benchmark PRIVATE
        NAVATALA_GPU_ROCM_VERSION=${NAVATALA_GPU_ROCM_VERSION})
endif()
if(NAVATALA_GPU_BUILD_MFMA_BENCHMARK)
    target_compile_definitions(rocm_vendor_benchmark PRIVATE NAVATALA_HAVE_MFMA_BENCHMARK=1)
else()
    target_compile_definitions(rocm_vendor_benchmark PRIVATE NAVATALA_HAVE_MFMA_BENCHMARK=0)
endif()

if(TARGET gpu_runtime)
    target_link_libraries(rocm_vendor_benchmark PRIVATE gpu_runtime)
    target_compile_definitions(rocm_vendor_benchmark PRIVATE NAVATALA_HAVE_RUNTIME_WRAPPER=1)
else()
    target_compile_definitions(rocm_vendor_benchmark PRIVATE NAVATALA_HAVE_RUNTIME_WRAPPER=0)
endif()

if(HIPSPARSELT_INCLUDE_DIR AND HIPSPARSELT_LIBRARY)
    target_include_directories(rocm_vendor_benchmark PRIVATE "${HIPSPARSELT_INCLUDE_DIR}")
    target_link_libraries(rocm_vendor_benchmark PRIVATE "${HIPSPARSELT_LIBRARY}")
    target_compile_definitions(rocm_vendor_benchmark PRIVATE NAVATALA_HAVE_HIPSPARSELT=1)
    message(STATUS "[Navatala GPU] hipSPARSELt detected for benchmark capability reporting")
else()
    target_compile_definitions(rocm_vendor_benchmark PRIVATE NAVATALA_HAVE_HIPSPARSELT=0)
    message(STATUS "[Navatala GPU] hipSPARSELt not found; benchmark will report it unavailable")
endif()

# The project scripts run this benchmark binary directly and keep the runtime
# test suite disabled. NAVATALA_GPU_BUILD_BENCHMARK_TESTS registers only these
# cheap benchmark smoke tests with CTest.
if(BUILD_TESTING AND NAVATALA_GPU_BUILD_BENCHMARK_TESTS)
    add_test(NAME rocm_vendor_benchmark_smoke
        COMMAND rocm_vendor_benchmark --quick --iterations 5 --json rocm_vendor_benchmark_smoke.json)
    if(TARGET gpu_runtime)
        add_test(NAME rocm_vendor_benchmark_wrapper_gemm_smoke
            COMMAND rocm_vendor_benchmark --quick --iterations 5
                    --include-wrapper-gemm-benchmark
                    --json rocm_vendor_benchmark_wrapper_gemm_smoke.json)
    endif()
endif()
