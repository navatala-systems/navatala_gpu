#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd

"""Validate a Navatala GPU Metal validation artifact.

The artifact is emitted by ``scripts/run_metal_validation.sh`` on an Apple
Silicon host. This validator is intentionally hardware-free: it checks the
public JSON contract, pass/fail status, and runtime-profile counter shape so a
captured Mac run can be reviewed on any development machine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "navatala_gpu.metal_validation.v1"
EXPECTED_TIMING_MODE = "ctest_runtime_profile"

REQUIRED_PROFILE_COUNTERS = {
    "submit",
    "sync",
    "command_buffer",
    "compute_encoder",
    "blit_encoder",
    "h2d_count",
    "h2d_bytes",
    "d2h_count",
    "d2h_bytes",
    "d2d_count",
    "d2d_bytes",
    "host_visible_copy",
    "batch_dispatch",
    "batch_flush",
    "batch_limit_flush",
    "skipped_empty_sync",
}

KNOWN_RUNS = {"baseline", "batched", "private"}


class ValidationError(Exception):
    """Raised when a Metal validation report violates the public contract."""


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


def _require_str(obj: dict[str, Any], key: str, path: str, *, allow_empty: bool = False) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        _fail(f"{path}.{key}", "expected non-empty string" if not allow_empty else "expected string")
    return value


def _require_bool(obj: dict[str, Any], key: str, path: str) -> bool:
    value = obj.get(key)
    if not isinstance(value, bool):
        _fail(f"{path}.{key}", "expected boolean")
    return value


def _require_int(obj: dict[str, Any], key: str, path: str, *, minimum: int | None = None) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(f"{path}.{key}", "expected integer")
    if minimum is not None and value < minimum:
        _fail(f"{path}.{key}", f"expected >= {minimum}, got {value}")
    return value


def _require_number(obj: dict[str, Any], key: str, path: str, *, minimum: float | None = None) -> float:
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
) -> list[str]:
    """Validate a decoded Metal validation report and return warnings."""

    warnings: list[str] = []

    schema = _require_str(report, "schema", "$")
    if schema != EXPECTED_SCHEMA:
        _fail("$.schema", f"expected {EXPECTED_SCHEMA!r}, got {schema!r}")

    timing_mode = _require_str(report, "timingMode", "$")
    if timing_mode != EXPECTED_TIMING_MODE:
        _fail("$.timingMode", f"expected {EXPECTED_TIMING_MODE!r}, got {timing_mode!r}")

    _require_str(report, "generatedAt", "$")
    _require_str(report, "commit", "$", allow_empty=True)

    host = _require_object(report.get("host"), "$.host")
    _require_str(host, "os", "$.host")
    _require_str(host, "arch", "$.host")
    _require_str(host, "macOS", "$.host", allow_empty=True)
    _require_str(host, "xcode", "$.host", allow_empty=True)
    _require_str(host, "deviceSummary", "$.host", allow_empty=True)

    build = _require_object(report.get("build"), "$.build")
    _require_str(build, "buildDir", "$.build")
    _require_str(build, "cmakeVersion", "$.build", allow_empty=True)
    _require_str(build, "compiler", "$.build", allow_empty=True)
    _require_str(build, "configuration", "$.build")

    tests = _require_object(report.get("tests"), "$.tests")
    _require_str(tests, "regex", "$.tests")

    runs = _require_list(report.get("runs"), "$.runs")
    if not runs:
        _fail("$.runs", "expected at least one run")

    seen_runs: set[str] = set()
    for idx, raw in enumerate(runs):
        path = f"$.runs[{idx}]"
        run = _require_object(raw, path)
        name = _require_str(run, "name", path)
        if name not in KNOWN_RUNS:
            _fail(f"{path}.name", f"unknown run name {name!r}")
        if name in seen_runs:
            _fail(f"{path}.name", f"duplicate run {name!r}")
        seen_runs.add(name)

        flags = _require_object(run.get("flags"), f"{path}.flags")
        _require_bool(flags, "batchSubmits", f"{path}.flags")
        _require_int(flags, "batchLimit", f"{path}.flags", minimum=0)
        _require_bool(flags, "privateDeviceBuffers", f"{path}.flags")

        ctest_passed = _require_bool(run, "ctestPassed", path)
        exit_code = _require_int(run, "exitCode", path, minimum=0)
        _require_number(run, "wallSeconds", path, minimum=0.0)
        _require_str(run, "log", path)
        _require_str(run, "notes", path, allow_empty=True)

        if ctest_passed and exit_code != 0:
            _fail(f"{path}.exitCode", "ctestPassed=true requires exitCode=0")
        if not allow_failures and not ctest_passed:
            _fail(f"{path}.ctestPassed", "ctestPassed=false")

        profile = _require_object(run.get("profileCounters"), f"{path}.profileCounters")
        missing_counters = sorted(REQUIRED_PROFILE_COUNTERS.difference(profile))
        if missing_counters:
            _fail(f"{path}.profileCounters", "missing counters: " + ", ".join(missing_counters))
        for key, value in profile.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                _fail(f"{path}.profileCounters.{key}", "expected non-negative integer")

        if name == "batched":
            if not flags["batchSubmits"]:
                _fail(f"{path}.flags.batchSubmits", "batched run must enable submit batching")
            if profile.get("batch_dispatch", 0) == 0:
                warnings.append("batched run has batch_dispatch=0; workload may not exercise batching")
        if name == "private" and not flags["privateDeviceBuffers"]:
            _fail(f"{path}.flags.privateDeviceBuffers", "private run must enable private device buffers")
        if profile.get("command_buffer", 0) == 0 and profile.get("submit", 0) > 0:
            warnings.append(f"{name}: submit count is non-zero but command_buffer=0")

    if require_full:
        missing_runs = sorted(KNOWN_RUNS.difference(seen_runs))
        if missing_runs:
            _fail("$.runs", "full validation requires runs: " + ", ".join(missing_runs))

    if "baseline" not in seen_runs:
        _fail("$.runs", "baseline run is required")

    baseline = next((run for run in runs if isinstance(run, dict) and run.get("name") == "baseline"), None)
    batched = next((run for run in runs if isinstance(run, dict) and run.get("name") == "batched"), None)
    if isinstance(baseline, dict) and isinstance(batched, dict):
        base_buffers = baseline.get("profileCounters", {}).get("command_buffer")
        batch_buffers = batched.get("profileCounters", {}).get("command_buffer")
        if isinstance(base_buffers, int) and isinstance(batch_buffers, int) and batch_buffers > base_buffers:
            warnings.append("batched run used more command buffers than baseline; investigate batching policy")

    return warnings


def load_report(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc
    return _require_object(data, "$")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path, help="Path to metal_validation.json")
    parser.add_argument("--allow-failures", action="store_true", help="Allow ctestPassed=false rows")
    parser.add_argument("--require-full", action="store_true", help="Require baseline, batched, and private runs")
    args = parser.parse_args(argv)

    try:
        report = load_report(args.json)
        warnings = validate_report(
            report,
            allow_failures=args.allow_failures,
            require_full=args.require_full,
        )
    except ValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"PASS: Metal validation artifact {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
