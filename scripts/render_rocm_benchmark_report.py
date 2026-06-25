#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd

"""Render a Markdown ROCm benchmark report from benchmark JSON."""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Any

from validate_rocm_benchmark_json import ValidationError, load_report, validate_report


def _markdown_escape(value: object) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _fmt_ms(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def _fmt_ratio(value: object) -> str:
    try:
        return f"{float(value):.3f}x"
    except (TypeError, ValueError):
        return ""


def _status(value: object) -> str:
    return "pass" if value is True else "fail"


def render_report(
    report: dict[str, Any],
    *,
    source_json: Path,
    validation_warnings: list[str],
) -> str:
    quick_mode = bool(report["quickMode"])
    generated_at = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    device = report["device"]

    lines: list[str] = [
        "# ROCm Vendor Benchmark Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Source JSON: `{source_json.name}`",
        f"- Schema: `{report['schema']}`",
        f"- Timing mode: `{report['timingMode']}`",
        f"- Quick mode: `{str(quick_mode).lower()}`",
        f"- Matrix set: `{report['matrix']}`",
        f"- Iterations: `{report['iterations']}`",
        f"- Warmup: `{report['warmup']}`",
        f"- Device: `{device['name']}`",
        f"- GCN arch: `{device['gcnArch']}`",
        f"- Global memory: `{device['globalMemoryMiB']} MiB`",
        f"- hipSPARSELt available: `{str(report['hipSPARSELtAvailable']).lower()}`",
        f"- hipSPARSELt mode: `{report['hipSPARSELtMode']}`",
        "",
    ]

    if quick_mode:
        lines.extend(
            [
                "> Quick mode is a smoke check only. Do not quote quick-mode numbers as performance claims.",
                "",
            ]
        )

    lines.extend(
        [
            "## Results",
            "",
            "| Operation | Shape | Generated path | Vendor baseline | Kernel class | Tuning path | Correctness | Generated mean ms | Vendor mean ms | Ratio | Max abs error |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["results"]:
        kernel_class = row.get("kernelClass") or (
            "vendor_library"
            if row.get("implementationKind") == "vendor_library" or str(row.get("operation", "")).startswith("HIPSPARSELT_")
            else "scalar"
        )
        tuning_path = row.get("tuningPath") or row.get("implementationKind") or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(row["operation"]),
                    _markdown_escape(row["shape"]),
                    _markdown_escape(row["navatalaPath"]),
                    _markdown_escape(row["vendorPath"]),
                    _markdown_escape(kernel_class),
                    _markdown_escape(tuning_path),
                    _status(row["correctness"]),
                    _fmt_ms(row["generatedMeanMs"]),
                    _fmt_ms(row["vendorMeanMs"]),
                    _fmt_ratio(row["generatedOverVendorRatio"]),
                    f"{float(row['maxAbsError']):.6g}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.",
            "- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.",
            "- This report covers only the operations listed above; it is not a full HIP backend certification.",
        ]
    )

    if report["hipSPARSELtMode"] == "capability_reporting_only":
        lines.append("- hipSPARSELt is reported as a capability only; no hipSPARSELt performance baseline is claimed.")

    diagnostic_rows = [
        row
        for row in report["results"]
        if "hostSubmitMeanMs" in row or "hostWallMeanMs" in row
    ]
    if diagnostic_rows:
        lines.extend(
            [
                "",
                "## Wrapper Timing Diagnostics",
                "",
                "| Operation | Shape | Stream-event mean ms | Host submit mean ms | Host wall mean ms | Submit/event | Wall/event |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in diagnostic_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_escape(row["operation"]),
                        _markdown_escape(row["shape"]),
                        _fmt_ms(row["generatedMeanMs"]),
                        _fmt_ms(row.get("hostSubmitMeanMs")),
                        _fmt_ms(row.get("hostWallMeanMs")),
                        _fmt_ratio(row.get("hostSubmitOverGeneratedRatio")),
                        _fmt_ratio(row.get("hostWallOverGeneratedRatio")),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "- `hostSubmitMeanMs` is CPU wall time to call the wrapper repeatedly and enqueue work, excluding the final event wait.",
                "- `hostWallMeanMs` includes the final event wait. Compare both with the stream-event mean to separate host submission overhead from device work.",
            ]
        )

    has_hipsparselt_row = any(str(row.get("operation", "")).startswith("HIPSPARSELT_") for row in report["results"])
    has_mfma_row = any(
        "mfma" in str(row.get("kernelClass", "")).lower()
        or "MFMA" in str(row.get("operation", ""))
        or "mfma" in str(row.get("navatalaPath", "")).lower()
        for row in report["results"]
    )
    if has_hipsparselt_row and has_mfma_row:
        lines.append(
            "- hipSPARSELt structured rows compute 50% structured sparse matmul; "
            "dense MFMA rows compute ordinary dense matmul. Do not compare their "
            "ratios directly."
        )

    if validation_warnings:
        lines.extend(["", "## Validation Warnings", ""])
        for warning in validation_warnings:
            lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path, help="Path to rocm_vendor_benchmark.json")
    parser.add_argument("--output", "-o", type=Path, help="Path to write Markdown report")
    parser.add_argument("--allow-failures", action="store_true", help="Allow correctness=false rows")
    parser.add_argument("--require-full", action="store_true", help="Reject quick-mode reports")
    parser.add_argument("--require-hipsparselt", action="store_true", help="Require hipSPARSELt availability")
    args = parser.parse_args(argv)

    try:
        report = load_report(args.json)
        warnings = validate_report(
            report,
            allow_failures=args.allow_failures,
            require_full=args.require_full,
            require_hipsparselt=args.require_hipsparselt,
        )
    except ValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    markdown = render_report(report, source_json=args.json, validation_warnings=warnings)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote ROCm benchmark report: {args.output}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
