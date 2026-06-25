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
        NAME rocm_benchmark_wrapper_mfma_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_mfma_packed_params_standard/rocm_vendor_benchmark.json"
            "--require-full")
    add_test(
        NAME rocm_benchmark_wrapper_semantics_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_semantics/rocm_vendor_benchmark.json"
            "--require-full")
    add_test(
        NAME rocm_benchmark_broad_mfma_hipsparselt_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_broad_wrapper_mfma_hipsparselt/rocm_vendor_benchmark.json"
            "--require-full"
            "--require-hipsparselt")
    add_test(
        NAME rocm_benchmark_broad_mfma_profile_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_broad_wrapper_mfma_hipsparselt/profile/rocm_vendor_benchmark.json"
            "--require-full")
    add_test(
        NAME rocm_benchmark_vendor_wrapper_cache_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_vendor_wrapper_cache/rocm_vendor_benchmark.json"
            "--require-full")
    add_test(
        NAME rocm_benchmark_wrapper_auto_vendor_dispatch_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_auto_vendor_dispatch/rocm_vendor_benchmark.json"
            "--require-full")
    add_test(
        NAME rocm_benchmark_wrapper_semantic_fulltile_50iter_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_semantic_fulltile_50iter/rocm_vendor_benchmark.json"
            "--require-full")
    add_test(
        NAME rocm_benchmark_edge_tails_profile_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_edge_tails_profile/rocm_vendor_benchmark.json"
            "--require-full")
    add_test(
        NAME rocm_benchmark_edge_tails_predicate_hoist_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260625_mi300x_edge_tails_predicate_hoist_rocm724/rocm_vendor_benchmark.json"
            "--require-full")
    add_test(
        NAME rocm_benchmark_edge_split_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260625_mi300x_edge_split_rocm724/rocm_vendor_benchmark.json"
            "--require-full")
    add_test(
        NAME rocm_benchmark_edge_nn_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260625_mi300x_edge_nn_rocm724/rocm_vendor_benchmark.json"
            "--require-full")
    add_test(
        NAME rocm_benchmark_edge_nn_profile_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260625_mi300x_edge_nn_rocm724/profile_edge_tails/rocm_vendor_benchmark.json"
            "--require-full")
    add_test(
        NAME rocm_benchmark_cta128_after_cache_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_rocm_benchmark_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_cta128_evidence_after_cache/rocm_vendor_benchmark.json"
            "--require-full")
    add_test(
        NAME rocm_benchmark_report_fixture_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/rocm_vendor_benchmark.sample.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark.sample.md")
    add_test(
        NAME metal_validation_sample_conformance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_metal_validation_json.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/metal_validation.sample.json"
            "--require-full")
    add_test(
        NAME metal_validation_sample_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_metal_validation_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/metal_validation.sample.json"
            "--require-full"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/metal_validation.sample.md")
    add_test(
        NAME rocm_benchmark_mi300x_reference_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/rocm_vendor_benchmark_mi300x_20260619.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_mi300x_20260619.md")
    add_test(
        NAME rocm_benchmark_wrapper_mfma_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_mfma_packed_params_standard/rocm_vendor_benchmark.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_wrapper_mfma.md")
    add_test(
        NAME rocm_benchmark_wrapper_semantics_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_semantics/rocm_vendor_benchmark.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_wrapper_semantics.md")
    add_test(
        NAME rocm_benchmark_vendor_wrapper_cache_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_vendor_wrapper_cache/rocm_vendor_benchmark.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_vendor_wrapper_cache.md")
    add_test(
        NAME rocm_benchmark_wrapper_auto_vendor_dispatch_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_auto_vendor_dispatch/rocm_vendor_benchmark.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_wrapper_auto_vendor_dispatch.md")
    add_test(
        NAME rocm_benchmark_wrapper_semantic_fulltile_50iter_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_semantic_fulltile_50iter/rocm_vendor_benchmark.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_wrapper_semantic_fulltile_50iter.md")
    add_test(
        NAME rocm_benchmark_edge_tails_profile_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_edge_tails_profile/rocm_vendor_benchmark.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_edge_tails_profile.md")
    add_test(
        NAME rocm_benchmark_edge_tails_predicate_hoist_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260625_mi300x_edge_tails_predicate_hoist_rocm724/rocm_vendor_benchmark.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_edge_tails_predicate_hoist.md")
    add_test(
        NAME rocm_benchmark_edge_split_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260625_mi300x_edge_split_rocm724/rocm_vendor_benchmark.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_edge_split.md")
    add_test(
        NAME rocm_benchmark_cta128_after_cache_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260624_mi300x_cta128_evidence_after_cache/rocm_vendor_benchmark.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_cta128_after_cache.md")
    add_test(
        NAME rocm_benchmark_edge_nn_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260625_mi300x_edge_nn_rocm724/rocm_vendor_benchmark.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_edge_nn.md")
    add_test(
        NAME rocm_benchmark_edge_nn_profile_render
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/render_rocm_benchmark_report.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/fixtures/hardware_runs/20260625_mi300x_edge_nn_rocm724/profile_edge_tails/rocm_vendor_benchmark.json"
            "--output"
            "${CMAKE_CURRENT_BINARY_DIR}/rocm_vendor_benchmark_edge_nn_profile.md")
    set(_navatala_gpu_python_tests
        manifest_conformance
        rocm_benchmark_json_fixture_conformance
        rocm_benchmark_mi300x_reference_conformance
        rocm_benchmark_wrapper_mfma_conformance
        rocm_benchmark_wrapper_semantics_conformance
        rocm_benchmark_broad_mfma_hipsparselt_conformance
        rocm_benchmark_broad_mfma_profile_conformance
        rocm_benchmark_vendor_wrapper_cache_conformance
        rocm_benchmark_wrapper_auto_vendor_dispatch_conformance
        rocm_benchmark_wrapper_semantic_fulltile_50iter_conformance
        rocm_benchmark_edge_tails_profile_conformance
        rocm_benchmark_edge_tails_predicate_hoist_conformance
        rocm_benchmark_edge_split_conformance
        rocm_benchmark_edge_nn_conformance
        rocm_benchmark_edge_nn_profile_conformance
        rocm_benchmark_cta128_after_cache_conformance
        rocm_benchmark_report_fixture_render
        metal_validation_sample_conformance
        metal_validation_sample_render
        rocm_benchmark_mi300x_reference_render
        rocm_benchmark_wrapper_mfma_render
        rocm_benchmark_wrapper_semantics_render
        rocm_benchmark_vendor_wrapper_cache_render
        rocm_benchmark_wrapper_auto_vendor_dispatch_render
        rocm_benchmark_wrapper_semantic_fulltile_50iter_render
        rocm_benchmark_edge_tails_profile_render
        rocm_benchmark_edge_tails_predicate_hoist_render
        rocm_benchmark_edge_split_render
        rocm_benchmark_cta128_after_cache_render
        rocm_benchmark_edge_nn_render
        rocm_benchmark_edge_nn_profile_render)
    foreach(_navatala_gpu_python_test IN LISTS _navatala_gpu_python_tests)
        set_tests_properties(
            "${_navatala_gpu_python_test}"
            PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    endforeach()
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
