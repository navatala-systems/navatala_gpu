#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd

"""No-GPU conformance checks for the published kernel corpus.

This is intentionally a release-shape test, not a numerical certification. It
asserts that the public manifest is internally consistent, that advertised
source/binary files exist, and that representative alpha operation families are
present for navigation by downstream users.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


EXPECTED_BACKENDS = {"cuda", "hip", "metal", "opencl", "vulkan"}
ALLOWED_LIBRARY_IMPLEMENTATIONS = {
    "library_contract",
    "vendor_dispatch",
    "fallback_reference",
    "unsupported",
}
REQUIRED_FAMILY_PREFIXES = {
    "samples": "navatala_samples_",
    "dense linear algebra": "navatala_linalg_",
    "sparse linear algebra": "navatala_sparse_",
    "dataframe": "navatala_dataframe_",
    "graph": "navatala_graph_",
    "ml/dnn": "navatala_ml_",
    "cfd": "navatala_cfd_",
}
REQUIRED_PUBLIC_OPS = {
    "navatala_samples_float32_add",
    "navatala_linalg_transpose_f32",
    "navatala_sparse_axpy_f32",
    "navatala_sparse_csr_spmv_row_f32",
}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    fail(f"expected boolean availability flag, got {value!r}")


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1]
    manifest_path = root / "kernels" / "manifest.json"
    if not manifest_path.exists():
        fail(f"missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernels = manifest.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        fail("manifest must contain a non-empty kernels list")

    summary = manifest.get("summary", {})
    declared_count = summary.get("kernelCount")
    if declared_count is not None and declared_count != len(kernels):
        fail(f"summary.kernelCount={declared_count} but kernels list has {len(kernels)}")
    library_ops = manifest.get("libraryOperations", [])
    if not isinstance(library_ops, list):
        fail("manifest.libraryOperations must be a list")
    declared_library_count = summary.get("libraryOperationCount")
    if declared_library_count is not None and declared_library_count != len(library_ops):
        fail(
            "summary.libraryOperationCount="
            f"{declared_library_count} but libraryOperations list has {len(library_ops)}"
        )

    seen_names: set[str] = set()
    seen_library_ids: set[str] = set()
    family_hits = {family: 0 for family in REQUIRED_FAMILY_PREFIXES}
    backend_hits = {backend: 0 for backend in EXPECTED_BACKENDS}
    blas_gemm_vendor_dispatch_seen = False

    for idx, kernel in enumerate(kernels):
        if not isinstance(kernel, dict):
            fail(f"kernel[{idx}] is not an object")
        name = kernel.get("name")
        if not isinstance(name, str) or not name:
            fail(f"kernel[{idx}] has no public name")
        if name in seen_names:
            fail(f"duplicate kernel name: {name}")
        seen_names.add(name)

        backends = kernel.get("backends")
        if not isinstance(backends, dict):
            fail(f"{name}: backends must be an object")
        missing_backends = EXPECTED_BACKENDS.difference(backends)
        if missing_backends:
            fail(f"{name}: missing backend entries {sorted(missing_backends)}")

        for backend in sorted(EXPECTED_BACKENDS):
            entry = backends[backend]
            if not isinstance(entry, dict):
                fail(f"{name}/{backend}: backend entry is not an object")
            if not as_bool(entry.get("available")):
                continue
            backend_hits[backend] += 1

            source = entry.get("source")
            if not isinstance(source, str) or not source:
                fail(f"{name}/{backend}: available backend has no source path")
            source_path = root / "kernels" / source
            if not source_path.exists():
                fail(f"{name}/{backend}: advertised source does not exist: {source}")

            binary = entry.get("binary")
            if binary is not None:
                if not isinstance(binary, str) or not binary:
                    fail(f"{name}/{backend}: binary path must be null or non-empty string")
                binary_path = root / "kernels" / binary
                if not binary_path.exists():
                    fail(f"{name}/{backend}: advertised binary does not exist: {binary}")

        for family, prefix in REQUIRED_FAMILY_PREFIXES.items():
            if name.startswith(prefix):
                family_hits[family] += 1

    for idx, op in enumerate(library_ops):
        if not isinstance(op, dict):
            fail(f"libraryOperations[{idx}] is not an object")
        op_id = op.get("id")
        if not isinstance(op_id, str) or not op_id:
            fail(f"libraryOperations[{idx}] has no public id")
        if op_id in seen_library_ids:
            fail(f"duplicate library operation id: {op_id}")
        seen_library_ids.add(op_id)

        operation = op.get("operation")
        if not isinstance(operation, str) or not operation:
            fail(f"{op_id}: operation must be a non-empty string")

        implementation = op.get("implementation")
        if not isinstance(implementation, dict):
            fail(f"{op_id}: implementation must be an object")
        kind = implementation.get("kind")
        if kind not in ALLOWED_LIBRARY_IMPLEMENTATIONS:
            fail(f"{op_id}: unknown implementation kind {kind!r}")
        vendor_backed = implementation.get("vendorBacked")
        if not isinstance(vendor_backed, bool):
            fail(f"{op_id}: implementation.vendorBacked must be boolean")
        vendor_backends = implementation.get("vendorBackends")
        if not isinstance(vendor_backends, list):
            fail(f"{op_id}: implementation.vendorBackends must be a list")
        unknown_backends = set(vendor_backends).difference(EXPECTED_BACKENDS)
        if unknown_backends:
            fail(f"{op_id}: unknown vendor backends {sorted(unknown_backends)}")
        if vendor_backed and not vendor_backends:
            fail(f"{op_id}: vendorBacked=true requires at least one vendor backend")
        if not vendor_backed and vendor_backends:
            fail(f"{op_id}: vendorBacked=false cannot list vendor backends")

        public_wrappers = implementation.get("publicWrappers", [])
        if not isinstance(public_wrappers, list):
            fail(f"{op_id}: implementation.publicWrappers must be a list")
        if (
            operation == "blas_gemm"
            and kind == "vendor_dispatch"
            and vendor_backed
            and {"cuda", "hip"}.issubset(set(vendor_backends))
            and "navatala_gpu_gemm_f32" in public_wrappers
        ):
            blas_gemm_vendor_dispatch_seen = True

    missing_ops = sorted(REQUIRED_PUBLIC_OPS.difference(seen_names))
    if missing_ops:
        fail(f"required public operation kernels missing: {missing_ops}")
    if not blas_gemm_vendor_dispatch_seen:
        fail("BLAS_GEMM vendor-dispatch library operation metadata missing")

    missing_families = [family for family, count in family_hits.items() if count == 0]
    if missing_families:
        fail(f"representative family coverage missing: {missing_families}")

    empty_backends = [backend for backend, count in backend_hits.items() if count == 0]
    if empty_backends:
        fail(f"no available kernels for backends: {empty_backends}")

    print(
        "PASS: manifest conformance "
        f"({len(kernels)} kernels, backend hits={backend_hits}, families={family_hits})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
