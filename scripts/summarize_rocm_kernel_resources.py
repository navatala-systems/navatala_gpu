#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd

"""Summarize rocprofv3 kernel resource metadata.

The ROCm benchmark profile flow emits rocprofv3 ``*_kernel_trace.csv`` files.
Those files include the static kernel resource fields that matter for tuning
gates: LDS bytes, scratch bytes, VGPRs, accumulator VGPRs, SGPRs, workgroup
geometry, and grid geometry. This helper distills the trace into one row per
kernel name so dated benchmark fixtures can carry a compact, reviewable record.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


RESOURCE_FIELDS = [
    "LDS_Block_Size",
    "Scratch_Size",
    "VGPR_Count",
    "Accum_VGPR_Count",
    "SGPR_Count",
    "Workgroup_Size_X",
    "Workgroup_Size_Y",
    "Workgroup_Size_Z",
    "Grid_Size_X",
    "Grid_Size_Y",
    "Grid_Size_Z",
]


def _find_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*_kernel_trace.csv"))


def _int_field(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(value)


def _duration_ns(row: dict[str, str]) -> int:
    return _int_field(row, "End_Timestamp") - _int_field(row, "Start_Timestamp")


def summarize(path: Path, *, kernel_substring: str | None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for trace in _find_trace_files(path):
        with trace.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                kernel = row.get("Kernel_Name", "")
                if not kernel:
                    continue
                if kernel_substring and kernel_substring not in kernel:
                    continue
                grouped[kernel].append(row)

    out: list[dict[str, Any]] = []
    for kernel in sorted(grouped):
        rows = grouped[kernel]
        first = rows[0]
        durations = [_duration_ns(row) for row in rows]
        summary: dict[str, Any] = {
            "Kernel_Name": kernel,
            "Dispatch_Count": len(rows),
            "Avg_Duration_Ns": round(sum(durations) / len(durations), 3),
            "Min_Duration_Ns": min(durations),
            "Max_Duration_Ns": max(durations),
        }
        for field in RESOURCE_FIELDS:
            summary[field] = _int_field(first, field)
        out.append(summary)
    return out


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Kernel_Name",
        "Dispatch_Count",
        "LDS_Block_Size",
        "Scratch_Size",
        "VGPR_Count",
        "Accum_VGPR_Count",
        "SGPR_Count",
        "Workgroup_Size_X",
        "Workgroup_Size_Y",
        "Workgroup_Size_Z",
        "Grid_Size_X",
        "Grid_Size_Y",
        "Grid_Size_Z",
        "Avg_Duration_Ns",
        "Min_Duration_Ns",
        "Max_Duration_Ns",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="rocprofv3 trace CSV or directory containing *_kernel_trace.csv")
    parser.add_argument("--kernel-substring", help="Only summarize kernels whose name contains this substring")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    rows = summarize(args.path, kernel_substring=args.kernel_substring)
    if not rows:
        raise SystemExit("no matching kernel rows found")
    write_csv(rows, args.output)
    print(f"Wrote ROCm kernel resource summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
