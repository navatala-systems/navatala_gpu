#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
#
# Run the Metal runtime policy sweep used to choose local opt-in settings.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${NAVATALA_GPU_METAL_POLICY_BUILD_DIR:-$ROOT_DIR/build-metal-benchmarks}"
RESULT_DIR="${NAVATALA_GPU_METAL_POLICY_RESULT_DIR:-$ROOT_DIR/benchmarks/results/metal-runtime-policy-sweep}"
BATCH_LIMITS="${NAVATALA_GPU_METAL_POLICY_BATCH_LIMITS:-8 16 32 64 128}"
SIZES="${NAVATALA_GPU_METAL_POLICY_SIZES:-65536 131072 196608 262144 524288 1048576 4194304}"
POLICY_LIMIT="${NAVATALA_GPU_METAL_POLICY_LIMIT:-64}"
PRIVATE_MIN_BYTES="${NAVATALA_GPU_METAL_POLICY_PRIVATE_MIN_BYTES:-0}"
ITERATIONS="${NAVATALA_GPU_METAL_POLICY_ITERATIONS:-60}"
WARMUP="${NAVATALA_GPU_METAL_POLICY_WARMUP:-6}"
BURST="${NAVATALA_GPU_METAL_POLICY_BURST:-32}"
COPY_REPEATS="${NAVATALA_GPU_METAL_POLICY_COPY_REPEATS:-12}"
RESULT_JSON="$RESULT_DIR/metal_runtime_policy_sweep.json"
RESULT_REPORT="$RESULT_DIR/metal_runtime_policy_sweep.md"

usage() {
  cat <<'EOF'
Usage: run_metal_runtime_policy_sweep.sh [options]

Options:
  --build-dir PATH       Shared CMake build directory.
  --result-dir PATH      Result directory for sweep artifacts.
  --batch-limits LIST    Space-separated batch limits to sweep.
  --sizes LIST           Space-separated element counts to sweep.
  --policy-limit N       Batch/blit limit for the size sweep.
  --private-min-bytes N  Threshold for private Device buffers in size sweep.
  --iterations N         Timed outer iterations per benchmark pass.
  --warmup N             Warmup outer iterations per benchmark pass.
  --burst N              Kernel submits per burst iteration.
  --copy-repeats N       H2D+D2H roundtrips per copy iteration.
  -h, --help             Show this help.

The script calls run_metal_runtime_benchmarks.sh repeatedly and aggregates the
per-pass JSON files. It is a policy probe, not a replacement for release
generated-kernel benchmark fixtures.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="${2:?missing value for --build-dir}"
      shift 2
      ;;
    --build-dir=*)
      BUILD_DIR="${1#*=}"
      shift
      ;;
    --result-dir)
      RESULT_DIR="${2:?missing value for --result-dir}"
      RESULT_JSON="$RESULT_DIR/metal_runtime_policy_sweep.json"
      RESULT_REPORT="$RESULT_DIR/metal_runtime_policy_sweep.md"
      shift 2
      ;;
    --result-dir=*)
      RESULT_DIR="${1#*=}"
      RESULT_JSON="$RESULT_DIR/metal_runtime_policy_sweep.json"
      RESULT_REPORT="$RESULT_DIR/metal_runtime_policy_sweep.md"
      shift
      ;;
    --batch-limits)
      BATCH_LIMITS="${2:?missing value for --batch-limits}"
      shift 2
      ;;
    --batch-limits=*)
      BATCH_LIMITS="${1#*=}"
      shift
      ;;
    --sizes)
      SIZES="${2:?missing value for --sizes}"
      shift 2
      ;;
    --sizes=*)
      SIZES="${1#*=}"
      shift
      ;;
    --policy-limit)
      POLICY_LIMIT="${2:?missing value for --policy-limit}"
      shift 2
      ;;
    --policy-limit=*)
      POLICY_LIMIT="${1#*=}"
      shift
      ;;
    --private-min-bytes)
      PRIVATE_MIN_BYTES="${2:?missing value for --private-min-bytes}"
      shift 2
      ;;
    --private-min-bytes=*)
      PRIVATE_MIN_BYTES="${1#*=}"
      shift
      ;;
    --iterations)
      ITERATIONS="${2:?missing value for --iterations}"
      shift 2
      ;;
    --iterations=*)
      ITERATIONS="${1#*=}"
      shift
      ;;
    --warmup)
      WARMUP="${2:?missing value for --warmup}"
      shift 2
      ;;
    --warmup=*)
      WARMUP="${1#*=}"
      shift
      ;;
    --burst)
      BURST="${2:?missing value for --burst}"
      shift 2
      ;;
    --burst=*)
      BURST="${1#*=}"
      shift
      ;;
    --copy-repeats)
      COPY_REPEATS="${2:?missing value for --copy-repeats}"
      shift 2
      ;;
    --copy-repeats=*)
      COPY_REPEATS="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$RESULT_DIR"
BENCH_SCRIPT="$ROOT_DIR/scripts/run_metal_runtime_benchmarks.sh"

run_matrix() {
  local name="$1"
  local limit="$2"
  local elements="$3"
  local out_dir="$RESULT_DIR/$name"
  mkdir -p "$out_dir"
  echo "=== Metal policy sweep: $name (limit=$limit elements=$elements) ==="
  NAVATALA_GPU_METAL_BENCH_BUILD_DIR="$BUILD_DIR" \
  NAVATALA_GPU_METAL_BATCH_LIMIT="$limit" \
  NAVATALA_GPU_METAL_BATCH_BLIT_LIMIT="$limit" \
  NAVATALA_GPU_METAL_PRIVATE_MIN_BYTES="$PRIVATE_MIN_BYTES" \
    "$BENCH_SCRIPT" \
      --build-dir "$BUILD_DIR" \
      --result-dir "$out_dir" \
      --iterations "$ITERATIONS" \
      --warmup "$WARMUP" \
      --elements "$elements" \
      --burst "$BURST" \
      --copy-repeats "$COPY_REPEATS" \
      >"$RESULT_DIR/${name}.stdout" 2>&1
  sed -n '1,80p' "$RESULT_DIR/${name}.stdout"
}

for limit in $BATCH_LIMITS; do
  run_matrix "batch_limit_${limit}" "$limit" 1048576
done

for elements in $SIZES; do
  run_matrix "size_${elements}" "$POLICY_LIMIT" "$elements"
done

python3 - "$RESULT_DIR" "$RESULT_JSON" "$RESULT_REPORT" "$BATCH_LIMITS" "$SIZES" "$POLICY_LIMIT" "$PRIVATE_MIN_BYTES" <<'PY'
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
result_json = Path(sys.argv[2])
result_report = Path(sys.argv[3])
batch_limits = sys.argv[4].split()
sizes = sys.argv[5].split()
policy_limit = int(sys.argv[6])
private_min_bytes = int(sys.argv[7])


def load_case(name: str) -> dict:
    path = result_dir / name / "metal_runtime_benchmark.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "path": str(path.relative_to(result_dir)),
        "payload": payload,
    }


def row(run: dict, operation: str) -> dict:
    for item in run.get("rows", []):
        if item.get("operation") == operation:
            return item
    return {}


def run_by_name(payload: dict, name: str) -> dict:
    for run in payload.get("runs", []):
        if run.get("name") == name:
            return run
    return {}


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


batch_cases = [load_case(f"batch_limit_{limit}") for limit in batch_limits]
size_cases = [load_case(f"size_{size}") for size in sizes]

report = {
    "schema": "navatala_gpu.metal_runtime_policy_sweep.v1",
    "generatedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "commit": os.environ.get("NAVATALA_GPU_SOURCE_COMMIT", ""),
    "policyLimit": policy_limit,
    "privateMinBytes": private_min_bytes,
    "batchLimits": [int(x) for x in batch_limits],
    "sizes": [int(x) for x in sizes],
    "batchSweep": [
        {
            "limit": int(case["name"].split("_")[-1]),
            "path": case["path"],
            "runs": case["payload"].get("runs", []),
        }
        for case in batch_cases
    ],
    "sizeSweep": [
        {
            "elements": int(case["name"].split("_")[-1]),
            "bytes": int(case["name"].split("_")[-1]) * 4,
            "path": case["path"],
            "runs": case["payload"].get("runs", []),
        }
        for case in size_cases
    ],
}
result_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

lines: list[str] = []
lines.append("# Metal Runtime Policy Sweep")
lines.append("")
lines.append(f"- Schema: `{report['schema']}`")
lines.append(f"- Generated: `{report['generatedAt']}`")
lines.append(f"- Commit: `{report['commit'] or '-'}`")
lines.append(f"- Policy limit for size sweep: `{policy_limit}`")
lines.append(f"- Private-buffer minimum bytes: `{private_min_bytes}`")
lines.append("")
lines.append("## Batch-Limit Sweep")
lines.append("")
lines.append("| Limit | batch_submits command buffers | private blit encoders | submit_burst ms | private+blit copy GB/s |")
lines.append("|---:|---:|---:|---:|---:|")
for case in batch_cases:
    limit = int(case["name"].split("_")[-1])
    payload = case["payload"]
    batch = run_by_name(payload, "batch_submits")
    private_blit = run_by_name(payload, "private_batch_blits")
    batch_c = batch.get("profileCounters", {})
    blit_c = private_blit.get("profileCounters", {})
    batch_burst = row(batch, "submit_burst")
    blit_copy = row(private_blit, "copy_roundtrip")
    lines.append(
        f"| {limit} | {batch_c.get('command_buffer', '-')} | "
        f"{blit_c.get('blit_encoder', '-')} | {fmt(batch_burst.get('meanMs'))} | "
        f"{fmt(blit_copy.get('throughputGBps'), 1)} |"
    )

lines.append("")
lines.append("## Buffer-Size Sweep")
lines.append("")
lines.append("| Elements | Bytes | baseline copy GB/s | private copy GB/s | private+blit copy GB/s | private+blit copy ms |")
lines.append("|---:|---:|---:|---:|---:|---:|")
for case in size_cases:
    elements = int(case["name"].split("_")[-1])
    payload = case["payload"]
    baseline = run_by_name(payload, "baseline")
    private = run_by_name(payload, "private")
    private_blit = run_by_name(payload, "private_batch_blits")
    baseline_copy = row(baseline, "copy_roundtrip")
    private_copy = row(private, "copy_roundtrip")
    blit_copy = row(private_blit, "copy_roundtrip")
    lines.append(
        f"| {elements} | {elements * 4} | {fmt(baseline_copy.get('throughputGBps'), 1)} | "
        f"{fmt(private_copy.get('throughputGBps'), 1)} | "
        f"{fmt(blit_copy.get('throughputGBps'), 1)} | {fmt(blit_copy.get('meanMs'))} |"
    )

lines.append("")
lines.append("## Policy Reading")
lines.append("")
lines.append("- Batch/blit limit `32` reaches the command-buffer and blit-encoder plateau for this benchmark; `64` is a conservative opt-in default.")
lines.append("- Shared buffers remain preferable for very small transfers; private buffers with batched blits become attractive around 512 KiB and strongly favorable from roughly 1 MiB upward.")
lines.append("- Use `NAVATALA_GPU_METAL_PRIVATE_MIN_BYTES` to keep small runtime buffers shared while allowing larger buffers to use private storage.")
lines.append("- These are local runtime-policy probes. Keep the knobs opt-in until the regenerated public tree has release-corpus benchmark fixtures.")
lines.append("")
result_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "Wrote:"
echo "  $RESULT_JSON"
echo "  $RESULT_REPORT"
