#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd

"""Validate ROCm vendor benchmark JSON.

This is a no-GPU release-shape check for the artifact emitted by
``benchmarks/rocm_vendor_benchmark.hip``. It deliberately validates only the
public JSON contract and correctness flags; live performance/correctness still
requires an AMD ROCm runner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "navatala_gpu.rocm_vendor_benchmark.v1"
EXPECTED_TIMING_MODE = "back_to_back_throughput_mean_per_launch"
KNOWN_HIPSPARSELT_MODES = {
    "capability_reporting_only",
    "vendor_benchmark",
}
KNOWN_MATRIX_SETS = {
    "quick",
    "standard",
    "broad",
    "cta128-evidence",
    "cfd-fixture",
}
KNOWN_IMPLEMENTATION_KINDS = {
    "portable_kernel",
    "tuned_kernel",
    "vendor_library",
}
KNOWN_KERNEL_CLASSES = {
    "scalar",
    "mfma_f16",
    "wmma_tf32",
    "vendor_library",
}
KNOWN_TUNING_PATHS = {
    "",
    "thread_per_row",
    "subgroup_per_row",
    "portable_f16_tiled",
    "portable_f16_f32out_tiled",
    "adaptive",
    "vendor_dispatch",
    "hip_mfma_gfx942_32x32x8_f16_f32_k_loop",
    "hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct",
    "hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared",
    "hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier",
    "hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded",
    "hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined",
    "hip_mfma_gfx942_128x128x32_f16_f32_cta128",
    "hip_mfma_gfx942_wrapper_dispatch",
}
TUNING_METADATA_KEYS = {
    "implementationKind",
    "tuningPath",
    "spmvRowNnzThreshold",
    "vendorDispatchSelected",
}


class ValidationError(Exception):
    """Raised when a benchmark report violates the public JSON contract."""


def _fail(path: str, message: str) -> None:
    raise ValidationError(f"{path}: {message}")


def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, f"expected object, got {type(value).__name__}")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, f"expected list, got {type(value).__name__}")
    return value


def _require_str(obj: dict[str, Any], key: str, path: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        _fail(f"{path}.{key}", "expected non-empty string")
    return value


def _require_string_field(obj: dict[str, Any], key: str, path: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str):
        _fail(f"{path}.{key}", "expected string")
    return value


def _require_bool(obj: dict[str, Any], key: str, path: str) -> bool:
    value = obj.get(key)
    if not isinstance(value, bool):
        _fail(f"{path}.{key}", "expected boolean")
    return value


def _require_int(obj: dict[str, Any], key: str, path: str, *, minimum: int) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(f"{path}.{key}", "expected integer")
    if value < minimum:
        _fail(f"{path}.{key}", f"expected >= {minimum}, got {value}")
    return value


def _require_number(
    obj: dict[str, Any],
    key: str,
    path: str,
    *,
    minimum: float | None = None,
) -> float:
    value = obj.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{path}.{key}", "expected number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{path}.{key}", "expected finite number")
    if minimum is not None and result < minimum:
        _fail(f"{path}.{key}", f"expected >= {minimum}, got {result}")
    return result


def validate_report(
    report: dict[str, Any],
    *,
    allow_failures: bool = False,
    require_full: bool = False,
    require_hipsparselt: bool = False,
) -> list[str]:
    """Validate a decoded benchmark report and return non-fatal warnings."""

    warnings: list[str] = []

    schema = _require_str(report, "schema", "$")
    if schema != EXPECTED_SCHEMA:
        _fail("$.schema", f"expected {EXPECTED_SCHEMA!r}, got {schema!r}")

    timing_mode = _require_str(report, "timingMode", "$")
    if timing_mode != EXPECTED_TIMING_MODE:
        _fail("$.timingMode", f"expected {EXPECTED_TIMING_MODE!r}, got {timing_mode!r}")

    iterations = _require_int(report, "iterations", "$", minimum=1)
    _require_int(report, "warmup", "$", minimum=0)
    quick_mode = _require_bool(report, "quickMode", "$")
    matrix = _require_str(report, "matrix", "$")
    if matrix not in KNOWN_MATRIX_SETS:
        _fail("$.matrix", f"unknown matrix set {matrix!r}")
    if quick_mode and matrix != "quick":
        _fail("$.matrix", "quickMode=true reports must use matrix='quick'")
    if not quick_mode and matrix == "quick":
        _fail("$.matrix", "quick matrix requires quickMode=true")
    device = _require_object(report.get("device"), "$.device")
    _require_str(device, "name", "$.device")
    _require_str(device, "gcnArch", "$.device")
    _require_int(device, "globalMemoryMiB", "$.device", minimum=1)
    rocminfo = report.get("rocminfo")
    if rocminfo is None:
        warnings.append("rocminfo metadata is missing; hardware provenance is incomplete")
    else:
        rocminfo_obj = _require_object(rocminfo, "$.rocminfo")
        _require_str(rocminfo_obj, "gcnArch", "$.rocminfo")
        _require_int(rocminfo_obj, "hipRuntimeVersion", "$.rocminfo", minimum=0)
        _require_int(rocminfo_obj, "hipDriverVersion", "$.rocminfo", minimum=0)
        _require_string_field(rocminfo_obj, "summary", "$.rocminfo")
    hipsparselt_available = _require_bool(report, "hipSPARSELtAvailable", "$")
    hipsparselt_mode = _require_str(report, "hipSPARSELtMode", "$")
    if hipsparselt_mode not in KNOWN_HIPSPARSELT_MODES:
        _fail("$.hipSPARSELtMode", f"unknown mode {hipsparselt_mode!r}")

    if require_full and quick_mode:
        _fail("$.quickMode", "full-report gate requires quickMode=false")
    if require_hipsparselt and not hipsparselt_available:
        _fail("$.hipSPARSELtAvailable", "required hipSPARSELt is not available")
    if quick_mode:
        warnings.append("quickMode=true: smoke-only report; do not use as a performance claim")
    if hipsparselt_mode == "capability_reporting_only":
        warnings.append("hipSPARSELt is capability reporting only; no hipSPARSELt benchmark row is claimed")

    results = _require_list(report.get("results"), "$.results")
    if not results:
        _fail("$.results", "expected at least one result")

    seen_results: set[tuple[str, str]] = set()
    has_hipsparselt_row = False
    for idx, raw in enumerate(results):
        path = f"$.results[{idx}]"
        result = _require_object(raw, path)
        operation = _require_str(result, "operation", path)
        shape = _require_str(result, "shape", path)
        result_key = (operation, shape)
        if result_key in seen_results:
            _fail(f"{path}.operation", f"duplicate result {operation!r} for shape {shape!r}")
        seen_results.add(result_key)
        _require_str(result, "navatalaPath", path)
        vendor_path = _require_str(result, "vendorPath", path)
        correctness = _require_bool(result, "correctness", path)
        generated_ms = _require_number(result, "generatedMeanMs", path, minimum=0.0)
        vendor_ms = _require_number(result, "vendorMeanMs", path, minimum=0.0)
        ratio = _require_number(result, "generatedOverVendorRatio", path, minimum=0.0)
        _require_number(result, "maxAbsError", path, minimum=0.0)
        _require_string_field(result, "notes", path)
        kernel_class = result.get("kernelClass")
        if kernel_class is None:
            kernel_class = "vendor_library" if operation.startswith("HIPSPARSELT_") else "scalar"
            warnings.append(f"{operation}: kernelClass missing; assuming {kernel_class}")
        elif not isinstance(kernel_class, str) or not kernel_class:
            _fail(f"{path}.kernelClass", "expected non-empty string")
        elif kernel_class not in KNOWN_KERNEL_CLASSES:
            _fail(f"{path}.kernelClass", f"unknown kernel class {kernel_class!r}")
        if any(key in result for key in TUNING_METADATA_KEYS):
            missing = sorted(key for key in TUNING_METADATA_KEYS if key not in result)
            if missing:
                _fail(path, "incomplete tuning metadata: missing " + ", ".join(missing))
            implementation_kind = _require_str(result, "implementationKind", path)
            tuning_path = _require_string_field(result, "tuningPath", path)
            _require_int(result, "spmvRowNnzThreshold", path, minimum=0)
            _require_bool(result, "vendorDispatchSelected", path)
            if implementation_kind not in KNOWN_IMPLEMENTATION_KINDS:
                _fail(f"{path}.implementationKind", f"unknown implementation kind {implementation_kind!r}")
            if tuning_path not in KNOWN_TUNING_PATHS:
                _fail(f"{path}.tuningPath", f"unknown tuning path {tuning_path!r}")
            if tuning_path == "" and implementation_kind != "portable_kernel":
                _fail(f"{path}.tuningPath", "empty tuningPath is only valid for portable_kernel rows")
            if implementation_kind == "vendor_library" and kernel_class != "vendor_library":
                _fail(f"{path}.kernelClass", "vendor_library rows must set kernelClass='vendor_library'")
            if operation == "CSR_SPMV_F32" and tuning_path not in {"thread_per_row", "subgroup_per_row", "adaptive"}:
                _fail(f"{path}.tuningPath", "CSR_SPMV_F32 requires a SpMV tuning path")
        if operation.startswith("HIPSPARSELT_"):
            has_hipsparselt_row = True

        if "rocBLAS" not in vendor_path and "rocSPARSE" not in vendor_path and "hipSPARSELt" not in vendor_path:
            warnings.append(f"{operation}: vendorPath does not name a ROCm vendor library")

        if not allow_failures and not correctness:
            _fail(f"{path}.correctness", "correctness=false")

        if generated_ms == 0.0:
            warnings.append(f"{operation}: generatedMeanMs is zero; check timer resolution")
        if vendor_ms == 0.0:
            warnings.append(f"{operation}: vendorMeanMs is zero; check timer resolution")
        if vendor_ms > 0.0:
            expected_ratio = generated_ms / vendor_ms
            tolerance = max(1.0e-6, abs(expected_ratio) * 1.0e-4)
            if abs(ratio - expected_ratio) > tolerance:
                _fail(
                    f"{path}.generatedOverVendorRatio",
                    f"expected approximately generatedMeanMs/vendorMeanMs={expected_ratio:.8g}, got {ratio:.8g}",
                )

    if iterations < 10 and not quick_mode:
        warnings.append("iterations<10 with quickMode=false; report may be too noisy for publication")
    if hipsparselt_mode == "vendor_benchmark" and not has_hipsparselt_row:
        _fail("$.hipSPARSELtMode", "vendor_benchmark mode requires at least one HIPSPARSELT_* result row")
    if hipsparselt_mode == "capability_reporting_only" and has_hipsparselt_row:
        _fail("$.hipSPARSELtMode", "HIPSPARSELT_* rows require hipSPARSELtMode='vendor_benchmark'")

    return warnings


def load_report(path: Path) -> dict[str, Any]:
    try:
        return _require_object(json.loads(path.read_text(encoding="utf-8")), "$")
    except OSError as exc:
        raise ValidationError(f"{path}: could not read file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path, help="Path to rocm_vendor_benchmark.json")
    parser.add_argument("--allow-failures", action="store_true", help="Allow correctness=false rows")
    parser.add_argument("--require-full", action="store_true", help="Reject quick-mode reports")
    parser.add_argument("--require-hipsparselt", action="store_true", help="Require hipSPARSELt availability")
    args = parser.parse_args(argv)

    try:
        warnings = validate_report(
            load_report(args.json),
            allow_failures=args.allow_failures,
            require_full=args.require_full,
            require_hipsparselt=args.require_hipsparselt,
        )
    except ValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"WARN: {warning}", file=sys.stderr)
    print(f"PASS: ROCm benchmark JSON is valid: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
