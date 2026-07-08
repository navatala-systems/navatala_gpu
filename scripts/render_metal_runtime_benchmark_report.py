#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
"""Render a Markdown report for the Metal runtime benchmark matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_float(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _fmt_int(value: object) -> str:
    try:
        return f"{int(value)}"
    except Exception:
        return "-"


def _row_by_operation(run: dict, operation: str) -> dict:
    for row in run.get("rows", []):
        if row.get("operation") == operation:
            return row
    return {}


def _flag_text(flags: dict) -> str:
    enabled = []
    if flags.get("privateDeviceBuffers"):
        min_bytes = flags.get("privateMinBytes", 0)
        try:
            min_bytes = int(min_bytes)
        except Exception:
            min_bytes = 0
        if min_bytes > 0:
            enabled.append(f"private buffers(min={min_bytes}B)")
        else:
            enabled.append("private buffers")
    if flags.get("batchSubmits"):
        enabled.append(f"submit batching(limit={flags.get('batchLimit', '-')})")
    if flags.get("batchBlits"):
        enabled.append(f"blit batching(limit={flags.get('batchBlitLimit', '-')})")
    return ", ".join(enabled) if enabled else "baseline shared buffers"


def render(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# Metal Runtime Benchmark")
    lines.append("")
    lines.append(f"- Schema: `{payload.get('schema', '-')}`")
    lines.append(f"- Generated: `{payload.get('generatedAt', '-')}`")
    lines.append(f"- Commit: `{payload.get('commit', '-')}`")
    host = payload.get("host", {})
    if host:
        lines.append(
            f"- Host: `{host.get('macOS', '-')}` `{host.get('arch', '-')}`; "
            f"{host.get('deviceSummary', '-')}"
        )
    lines.append(f"- Mode: `{payload.get('mode', '-')}`")
    lines.append("")

    lines.append("## Runtime Counters")
    lines.append("")
    lines.append(
        "| Run | Flags | Pass | Command buffers | Compute encoders | "
        "Blit encoders | Host-visible copies | Batch dispatch | Batch flush | "
        "Archive hit/miss/store |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in payload.get("runs", []):
        counters = run.get("profileCounters", {})
        lines.append(
            "| {name} | {flags} | {passed} | {cmd} | {comp} | {blit} | "
            "{host_visible} | {batch_dispatch} | {batch_flush} | {archive} |".format(
                name=run.get("name", "-"),
                flags=_flag_text(run.get("flags", {})),
                passed="yes" if run.get("passed") else "no",
                cmd=_fmt_int(counters.get("command_buffer")),
                comp=_fmt_int(counters.get("compute_encoder")),
                blit=_fmt_int(counters.get("blit_encoder")),
                host_visible=_fmt_int(counters.get("host_visible_copy")),
                batch_dispatch=_fmt_int(counters.get("batch_dispatch")),
                batch_flush=_fmt_int(counters.get("batch_flush")),
                archive=(
                    f"{_fmt_int(counters.get('metal_archive_cache_hit'))}/"
                    f"{_fmt_int(counters.get('metal_archive_cache_miss'))}/"
                    f"{_fmt_int(counters.get('metal_archive_cache_store'))}"
                ),
            )
        )
    lines.append("")

    lines.append("## Timings")
    lines.append("")
    lines.append(
        "| Run | submit_sync mean ms | submit_burst mean ms | "
        "copy_roundtrip mean ms | copy_roundtrip GB/s |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for run in payload.get("runs", []):
        submit_sync = _row_by_operation(run, "submit_sync")
        submit_burst = _row_by_operation(run, "submit_burst")
        copy_roundtrip = _row_by_operation(run, "copy_roundtrip")
        lines.append(
            "| {name} | {sync_ms} | {burst_ms} | {copy_ms} | {copy_gbps} |".format(
                name=run.get("name", "-"),
                sync_ms=_fmt_float(submit_sync.get("meanMs")),
                burst_ms=_fmt_float(submit_burst.get("meanMs")),
                copy_ms=_fmt_float(copy_roundtrip.get("meanMs")),
                copy_gbps=_fmt_float(copy_roundtrip.get("throughputGBps"), digits=1),
            )
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This benchmark is a runtime-overhead probe. It uses inline MSL and the "
        "public `gpu_runtime` API, so it can run on a Metal-only generated tree. "
        "It does not replace generated-kernel throughput fixtures."
    )
    lines.append("")
    lines.append(
        "Use the counter deltas to validate command-buffer and blit-encoder "
        "optimizations; use timing rows as host-local evidence only, because "
        "Apple Silicon scheduling and thermal state can shift absolute timings."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path, help="Metal runtime benchmark JSON")
    parser.add_argument("--output", type=Path, help="Markdown output path")
    args = parser.parse_args()

    payload = json.loads(args.json.read_text(encoding="utf-8"))
    report = render(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
