#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
#
# Build and run the Metal runtime benchmark matrix.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${NAVATALA_GPU_METAL_BENCH_BUILD_DIR:-$ROOT_DIR/build-metal-benchmarks}"
RESULT_DIR="${NAVATALA_GPU_METAL_BENCH_RESULT_DIR:-$ROOT_DIR/benchmarks/results/metal-runtime}"
RESULT_JSON="$RESULT_DIR/metal_runtime_benchmark.json"
RESULT_REPORT="$RESULT_DIR/metal_runtime_benchmark.md"
if [[ -n "${NAVATALA_GPU_METAL_CACHE_HOME:-}" ]]; then
  CACHE_HOME="$NAVATALA_GPU_METAL_CACHE_HOME"
  CACHE_HOME_EXPLICIT=1
else
  CACHE_HOME="$RESULT_DIR/cache_home"
  CACHE_HOME_EXPLICIT=0
fi
QUICK=0
ELEMENTS="${NAVATALA_GPU_METAL_BENCH_ELEMENTS:-1048576}"
ITERATIONS="${NAVATALA_GPU_METAL_BENCH_ITERATIONS:-100}"
WARMUP="${NAVATALA_GPU_METAL_BENCH_WARMUP:-10}"
BURST="${NAVATALA_GPU_METAL_BENCH_BURST:-32}"
COPY_REPEATS="${NAVATALA_GPU_METAL_BENCH_COPY_REPEATS:-16}"
BATCH_LIMIT="${NAVATALA_GPU_METAL_BATCH_LIMIT:-64}"
BLIT_LIMIT="${NAVATALA_GPU_METAL_BATCH_BLIT_LIMIT:-64}"
PRIVATE_MIN_BYTES="${NAVATALA_GPU_METAL_PRIVATE_MIN_BYTES:-0}"

usage() {
  cat <<'EOF'
Usage: run_metal_runtime_benchmarks.sh [options]

Options:
  --build-dir PATH       CMake build directory.
  --result-dir PATH      Result directory for logs and JSON artifacts.
  --json PATH            Output aggregate JSON path.
  --quick                Use a smoke-sized benchmark.
  --elements N           Float element count.
  --iterations N         Timed outer iterations.
  --warmup N             Warmup outer iterations.
  --burst N              Kernel submits per burst iteration.
  --copy-repeats N       H2D+D2H roundtrips per copy iteration.
  -h, --help             Show this help.

Environment:
  NAVATALA_GPU_METAL_BATCH_LIMIT controls the submit-batching pass.
  NAVATALA_GPU_METAL_BATCH_BLIT_LIMIT controls the private-buffer blit pass.
  NAVATALA_GPU_METAL_PRIVATE_MIN_BYTES keeps smaller Device buffers in shared
  storage even when NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS=1. Default 0.
  NAVATALA_GPU_METAL_CACHE_HOME sets the isolated XDG cache root used for
  Metal binary-archive cache measurements. Defaults under --result-dir.
  NAVATALA_GPU_METAL_PRESERVE_CACHE=1 preserves the cache between invocations;
  by default the Metal cache under CACHE_HOME is cleared before the matrix.
  NAVATALA_GPU_SOURCE_COMMIT overrides the JSON commit field when the benchmark
  runs from a regenerated tree that is not itself a Git checkout.
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
      RESULT_JSON="$RESULT_DIR/metal_runtime_benchmark.json"
      RESULT_REPORT="$RESULT_DIR/metal_runtime_benchmark.md"
      if [[ "$CACHE_HOME_EXPLICIT" == "0" ]]; then
        CACHE_HOME="$RESULT_DIR/cache_home"
      fi
      shift 2
      ;;
    --result-dir=*)
      RESULT_DIR="${1#*=}"
      RESULT_JSON="$RESULT_DIR/metal_runtime_benchmark.json"
      RESULT_REPORT="$RESULT_DIR/metal_runtime_benchmark.md"
      if [[ "$CACHE_HOME_EXPLICIT" == "0" ]]; then
        CACHE_HOME="$RESULT_DIR/cache_home"
      fi
      shift
      ;;
    --json)
      RESULT_JSON="${2:?missing value for --json}"
      RESULT_REPORT="${RESULT_JSON%.*}.md"
      shift 2
      ;;
    --json=*)
      RESULT_JSON="${1#*=}"
      RESULT_REPORT="${RESULT_JSON%.*}.md"
      shift
      ;;
    --quick)
      QUICK=1
      shift
      ;;
    --elements)
      ELEMENTS="${2:?missing value for --elements}"
      shift 2
      ;;
    --elements=*)
      ELEMENTS="${1#*=}"
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

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: Metal benchmarks must run on macOS with Apple Metal support." >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
mkdir -p "$(dirname "$RESULT_JSON")"
mkdir -p "$CACHE_HOME"
if [[ "${NAVATALA_GPU_METAL_PRESERVE_CACHE:-0}" != "1" ]]; then
  rm -rf "$CACHE_HOME/gpu_runtime/metal"
fi

NPROC="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "=== Metal runtime benchmark environment ==="
sw_vers || true
xcodebuild -version || true
system_profiler SPDisplaysDataType 2>/dev/null | sed -n '1,80p' || true
cmake --version | sed -n '1p' || true
${CXX:-c++} --version 2>/dev/null | sed -n '1p' || true

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DNAVATALA_GPU_USE_METAL=ON \
  -DNAVATALA_GPU_USE_CUDA=OFF \
  -DNAVATALA_GPU_USE_HIP=OFF \
  -DNAVATALA_GPU_USE_VULKAN=OFF \
  -DNAVATALA_GPU_USE_OPENCL=OFF \
  -DNAVATALA_GPU_BUILD_RUNTIME=ON \
  -DNAVATALA_GPU_BUILD_EXAMPLES=OFF \
  -DNAVATALA_GPU_BUILD_TESTS=OFF \
  -DNAVATALA_GPU_BUILD_BENCHMARKS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" --target metal_runtime_benchmark -j"$NPROC"

BENCH="$BUILD_DIR/benchmarks/metal_runtime_benchmark"
if [[ ! -x "$BENCH" ]]; then
  echo "ERROR: benchmark binary not found: $BENCH" >&2
  exit 1
fi

BASE_ARGS=()
if [[ "$QUICK" == "1" ]]; then
  BASE_ARGS+=(--quick)
else
  BASE_ARGS+=(--elements "$ELEMENTS" --iterations "$ITERATIONS" --warmup "$WARMUP" --burst "$BURST" --copy-repeats "$COPY_REPEATS")
fi

run_pass() {
  local name="$1"
  shift
  local log="$RESULT_DIR/${name}.log"
  local json="$RESULT_DIR/${name}.json"

  echo "=== Metal runtime benchmark pass: $name ==="
  set +e
  /usr/bin/time -p env "$@" \
    GPU_RUNTIME_BACKEND=metal \
    XDG_CACHE_HOME="$CACHE_HOME" \
    NAVATALA_GPU_RUNTIME_PROFILE=1 \
    "$BENCH" "${BASE_ARGS[@]}" --json "$json" \
    >"$log" 2>&1
  local status=$?
  set -e
  sed -n '1,180p' "$log"
  echo "$status" >"$RESULT_DIR/${name}.exitcode"
}

run_pass baseline \
  NAVATALA_GPU_METAL_BATCH_SUBMITS=0 \
  NAVATALA_GPU_METAL_BATCH_BLITS=0 \
  NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS=0

run_pass batch_submits \
  NAVATALA_GPU_METAL_BATCH_SUBMITS=1 \
  NAVATALA_GPU_METAL_BATCH_LIMIT="$BATCH_LIMIT" \
  NAVATALA_GPU_METAL_BATCH_BLITS=0 \
  NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS=0

run_pass private \
  NAVATALA_GPU_METAL_BATCH_SUBMITS=0 \
  NAVATALA_GPU_METAL_BATCH_BLITS=0 \
  NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS=1 \
  NAVATALA_GPU_METAL_PRIVATE_MIN_BYTES="$PRIVATE_MIN_BYTES"

run_pass private_batch_blits \
  NAVATALA_GPU_METAL_BATCH_SUBMITS=0 \
  NAVATALA_GPU_METAL_BATCH_BLITS=1 \
  NAVATALA_GPU_METAL_BATCH_BLIT_LIMIT="$BLIT_LIMIT" \
  NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS=1 \
  NAVATALA_GPU_METAL_PRIVATE_MIN_BYTES="$PRIVATE_MIN_BYTES"

python3 - "$ROOT_DIR" "$BUILD_DIR" "$RESULT_DIR" "$RESULT_JSON" "$QUICK" "$BATCH_LIMIT" "$BLIT_LIMIT" "$PRIVATE_MIN_BYTES" "$CACHE_HOME" <<'PY'
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
build_dir = Path(sys.argv[2])
result_dir = Path(sys.argv[3])
result_json = Path(sys.argv[4])
quick = sys.argv[5] == "1"
batch_limit = int(sys.argv[6])
blit_limit = int(sys.argv[7])
private_min_bytes = int(sys.argv[8])
cache_home = sys.argv[9]

COUNTERS = {
    "submit": 0,
    "sync": 0,
    "command_buffer": 0,
    "compute_encoder": 0,
    "blit_encoder": 0,
    "h2d_count": 0,
    "h2d_bytes": 0,
    "d2h_count": 0,
    "d2h_bytes": 0,
    "d2d_count": 0,
    "d2d_bytes": 0,
    "hosthost_count": 0,
    "hosthost_bytes": 0,
    "host_visible_copy": 0,
    "batch_dispatch": 0,
    "batch_flush": 0,
    "batch_limit_flush": 0,
    "skipped_empty_sync": 0,
    "metal_archive_cache_hit": 0,
    "metal_archive_cache_miss": 0,
    "metal_archive_cache_store": 0,
    "metal_archive_cache_store_bytes": 0,
}


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return ""


def first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


def parse_profile(log_text: str) -> tuple[int, dict[str, int]]:
    counters = dict(COUNTERS)
    profile_lines = 0
    for line in log_text.splitlines():
        if "NAVATALA_GPU_RUNTIME_PROFILE:" not in line:
            continue
        profile_lines += 1
        for key, value in re.findall(r"([A-Za-z0-9_]+)=([0-9]+)", line):
            if key in counters:
                counters[key] += int(value)
    return profile_lines, counters


def parse_wall(log_text: str) -> float:
    wall = 0.0
    for line in log_text.splitlines():
        if line.startswith("real "):
            try:
                wall = float(line.split()[1])
            except Exception:
                pass
    return wall


def parse_run(name: str, flags: dict) -> dict:
    log = result_dir / f"{name}.log"
    bench_json = result_dir / f"{name}.json"
    exitcode_path = result_dir / f"{name}.exitcode"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    profile_lines, counters = parse_profile(text)
    exit_code = int(exitcode_path.read_text(encoding="utf-8").strip()) if exitcode_path.exists() else 1
    payload = {}
    if bench_json.exists():
        payload = json.loads(bench_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    rows_passed = bool(rows) and all(bool(row.get("passed")) for row in rows)
    return {
        "name": name,
        "flags": flags,
        "exitCode": exit_code,
        "passed": exit_code == 0 and rows_passed,
        "wallSeconds": parse_wall(text),
        "log": log.name,
        "benchmarkJson": bench_json.name,
        "profileLineCount": profile_lines,
        "profileCounters": counters,
        "rows": rows,
    }


def device_summary() -> str:
    raw = run_text(["/usr/sbin/system_profiler", "SPDisplaysDataType"])
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(("Chipset Model:", "Metal Support:", "Vendor:", "VRAM")):
            lines.append(stripped)
    return "; ".join(lines)


runs = [
    parse_run("baseline", {
        "batchSubmits": False,
        "batchBlits": False,
        "privateDeviceBuffers": False,
    }),
    parse_run("batch_submits", {
        "batchSubmits": True,
        "batchLimit": batch_limit,
        "batchBlits": False,
        "privateDeviceBuffers": False,
    }),
    parse_run("private", {
        "batchSubmits": False,
        "batchBlits": False,
        "privateDeviceBuffers": True,
        "privateMinBytes": private_min_bytes,
    }),
    parse_run("private_batch_blits", {
        "batchSubmits": False,
        "batchBlits": True,
        "batchBlitLimit": blit_limit,
        "privateDeviceBuffers": True,
        "privateMinBytes": private_min_bytes,
    }),
]

report = {
    "schema": "navatala_gpu.metal_runtime_benchmark_matrix.v1",
    "timingMode": "host_wall_synchronized_runtime_api",
    "generatedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "commit": os.environ.get("NAVATALA_GPU_SOURCE_COMMIT")
        or run_text(["git", "-C", str(root), "rev-parse", "HEAD"]),
    "host": {
        "os": platform.system(),
        "arch": platform.machine(),
        "macOS": run_text(["/usr/bin/sw_vers", "-productVersion"]),
        "xcode": first_line(run_text(["/usr/bin/xcodebuild", "-version"])),
        "deviceSummary": device_summary(),
    },
    "build": {
        "buildDir": str(build_dir),
        "cacheHome": cache_home,
        "cmakeVersion": first_line(run_text(["cmake", "--version"])),
        "compiler": first_line(run_text([os.environ.get("CXX", "c++"), "--version"])),
        "configuration": "Release",
    },
    "mode": "quick" if quick else "full",
    "runs": runs,
}

result_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

failed = [run["name"] for run in runs if not run["passed"]]
if failed:
    print(f"ERROR: benchmark run(s) failed: {', '.join(failed)}", file=sys.stderr)
    sys.exit(1)
PY

python3 "$ROOT_DIR/scripts/render_metal_runtime_benchmark_report.py" "$RESULT_JSON" --output "$RESULT_REPORT"

echo "Wrote:"
echo "  $RESULT_JSON"
echo "  $RESULT_REPORT"
for name in baseline batch_submits private private_batch_blits; do
  echo "  $RESULT_DIR/${name}.log"
  echo "  $RESULT_DIR/${name}.json"
done
