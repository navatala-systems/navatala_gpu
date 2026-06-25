#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd

"""Render a Markdown report from a Metal validation artifact."""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Any

from validate_metal_validation_json import ValidationError, load_report, validate_report


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _fmt_seconds(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _status(value: object) -> str:
    return "pass" if value is True else "fail"


def _counter(profile: dict[str, Any], key: str) -> int:
    value = profile.get(key, 0)
    return value if isinstance(value, int) else 0


def render_report(
    report: dict[str, Any],
    *,
    source_json: Path,
    validation_warnings: list[str],
) -> str:
    generated_at = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    host = report["host"]
    build = report["build"]

    lines: list[str] = [
        "# Metal Validation Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Source JSON: `{source_json.name}`",
        f"- Schema: `{report['schema']}`",
        f"- Timing mode: `{report['timingMode']}`",
        f"- Commit: `{report['commit']}`",
        f"- Host: `{host['os']} {host['macOS']} ({host['arch']})`",
        f"- Xcode: `{host['xcode']}`",
        f"- Device summary: `{host['deviceSummary']}`",
        f"- Build directory: `{build['buildDir']}`",
        f"- CMake: `{build['cmakeVersion']}`",
        f"- Compiler: `{build['compiler']}`",
        f"- Configuration: `{build['configuration']}`",
        "",
        "## Runs",
        "",
        "| Run | Batching | Private buffers | CTest | Wall s | submit | sync | command buffers | compute encoders | blit encoders | host-visible copies | batch dispatches | skipped empty syncs |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for run in report["runs"]:
        flags = run["flags"]
        profile = run["profileCounters"]
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(run["name"]),
                    str(flags["batchSubmits"]).lower(),
                    str(flags["privateDeviceBuffers"]).lower(),
                    _status(run["ctestPassed"]),
                    _fmt_seconds(run["wallSeconds"]),
                    str(_counter(profile, "submit")),
                    str(_counter(profile, "sync")),
                    str(_counter(profile, "command_buffer")),
                    str(_counter(profile, "compute_encoder")),
                    str(_counter(profile, "blit_encoder")),
                    str(_counter(profile, "host_visible_copy")),
                    str(_counter(profile, "batch_dispatch")),
                    str(_counter(profile, "skipped_empty_sync")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- This artifact is a runtime correctness and overhead probe, not a broad Metal performance benchmark.",
            "- `command_buffer`, `compute_encoder`, and `blit_encoder` come from `NAVATALA_GPU_RUNTIME_PROFILE=1`.",
            "- The batched run should reduce command-buffer pressure for command-buffer-bound workloads.",
            "- Private-buffer mode is experimental until Apple hardware runs show correctness and acceptable copy overhead.",
        ]
    )

    if validation_warnings:
        lines.extend(["", "## Validation Warnings", ""])
        for warning in validation_warnings:
            lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path, help="Path to metal_validation.json")
    parser.add_argument("--output", "-o", type=Path, help="Path to write Markdown report")
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

    markdown = render_report(report, source_json=args.json, validation_warnings=warnings)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote Metal validation report: {args.output}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
