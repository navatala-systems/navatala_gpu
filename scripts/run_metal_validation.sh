#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
#
# Build and run the Metal runtime validation pass on an Apple Silicon host.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${NAVATALA_GPU_METAL_BUILD_DIR:-$ROOT_DIR/build-metal-validation}"
RESULT_DIR="${NAVATALA_GPU_METAL_RESULT_DIR:-$ROOT_DIR/benchmarks/results/metal-validation}"
TEST_REGEX="${NAVATALA_GPU_METAL_TEST_REGEX:-test_device_selection|test_events|test_graphs|test_kernel_execution|test_program_cache|test_memory_resources|test_stream_pool|test_device_buffer|manifest_conformance|metal_validation_sample_conformance|metal_validation_sample_render}"
RUN_PRIVATE="${NAVATALA_GPU_METAL_RUN_PRIVATE:-1}"
RESULT_JSON="$RESULT_DIR/metal_validation.json"
RESULT_REPORT="$RESULT_DIR/metal_validation.md"

usage() {
  cat <<'EOF'
Usage: run_metal_validation.sh [options]

Options:
  --build-dir PATH       CMake build directory.
  --result-dir PATH      Result directory for logs and JSON/Markdown artifacts.
  --test-regex REGEX     CTest regular expression.
  --skip-private         Skip the private-device-buffer validation pass.
  --json PATH            Output JSON path.
  -h, --help             Show this help.

Environment:
  NAVATALA_GPU_METAL_BATCH_LIMIT defaults to 64 for the batched pass.
  NAVATALA_GPU_METAL_RUN_PRIVATE=0 is equivalent to --skip-private.
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
      RESULT_JSON="$RESULT_DIR/metal_validation.json"
      RESULT_REPORT="$RESULT_DIR/metal_validation.md"
      shift 2
      ;;
    --result-dir=*)
      RESULT_DIR="${1#*=}"
      RESULT_JSON="$RESULT_DIR/metal_validation.json"
      RESULT_REPORT="$RESULT_DIR/metal_validation.md"
      shift
      ;;
    --test-regex)
      TEST_REGEX="${2:?missing value for --test-regex}"
      shift 2
      ;;
    --test-regex=*)
      TEST_REGEX="${1#*=}"
      shift
      ;;
    --skip-private)
      RUN_PRIVATE=0
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
  echo "ERROR: Metal validation must run on macOS with Apple Metal support." >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
mkdir -p "$(dirname "$RESULT_JSON")"

NPROC="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
BATCH_LIMIT="${NAVATALA_GPU_METAL_BATCH_LIMIT:-64}"

echo "=== Metal validation environment ==="
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
  -DNAVATALA_GPU_BUILD_TESTS=ON \
  -DNAVATALA_GPU_BUILD_BENCHMARKS=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" -j"$NPROC"

run_pass() {
  local name="$1"
  local log="$RESULT_DIR/${name}.log"
  shift

  echo "=== Metal validation pass: $name ==="
  set +e
  /usr/bin/time -p env "$@" \
    GPU_RUNTIME_BACKEND=metal \
    NAVATALA_GPU_RUNTIME_PROFILE=1 \
    ctest --test-dir "$BUILD_DIR" --output-on-failure -R "$TEST_REGEX" \
    >"$log" 2>&1
  local status=$?
  set -e
  sed -n '1,220p' "$log"
  echo "$status" >"$RESULT_DIR/${name}.exitcode"
}

run_pass baseline \
  NAVATALA_GPU_METAL_BATCH_SUBMITS=0 \
  NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS=0

run_pass batched \
  NAVATALA_GPU_METAL_BATCH_SUBMITS=1 \
  NAVATALA_GPU_METAL_BATCH_LIMIT="$BATCH_LIMIT" \
  NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS=0

if [[ "$RUN_PRIVATE" == "1" ]]; then
  run_pass private \
    NAVATALA_GPU_METAL_BATCH_SUBMITS=0 \
    NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS=1
fi

python3 - "$ROOT_DIR" "$BUILD_DIR" "$RESULT_DIR" "$RESULT_JSON" "$TEST_REGEX" "$BATCH_LIMIT" "$RUN_PRIVATE" <<'PY'
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
test_regex = sys.argv[5]
batch_limit = int(sys.argv[6])
run_private = sys.argv[7] == "1"

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
}


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return ""


def first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


def parse_log(name: str) -> dict:
    log = result_dir / f"{name}.log"
    exitcode_path = result_dir / f"{name}.exitcode"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    counters = dict(COUNTERS)
    profile_lines = 0
    for line in text.splitlines():
        if "NAVATALA_GPU_RUNTIME_PROFILE:" not in line:
            continue
        profile_lines += 1
        for key, value in re.findall(r"([A-Za-z0-9_]+)=([0-9]+)", line):
            if key in counters:
                counters[key] += int(value)
    wall_seconds = 0.0
    for line in text.splitlines():
        if line.startswith("real "):
            try:
                wall_seconds = float(line.split()[1])
            except Exception:
                pass
    exit_code = int(exitcode_path.read_text(encoding="utf-8").strip()) if exitcode_path.exists() else 1
    flags = {
        "batchSubmits": name == "batched",
        "batchLimit": batch_limit if name == "batched" else 0,
        "privateDeviceBuffers": name == "private",
    }
    return {
        "name": name,
        "flags": flags,
        "ctestPassed": exit_code == 0,
        "exitCode": exit_code,
        "wallSeconds": wall_seconds,
        "log": log.name,
        "profileLineCount": profile_lines,
        "profileCounters": counters,
        "notes": "",
    }


runs = [parse_log("baseline"), parse_log("batched")]
if run_private:
    runs.append(parse_log("private"))

commit = run_text(["git", "-C", str(root), "rev-parse", "HEAD"])
device_summary = run_text(["/usr/sbin/system_profiler", "SPDisplaysDataType"])
device_lines = []
for line in device_summary.splitlines():
    stripped = line.strip()
    if stripped.startswith(("Chipset Model:", "Metal Support:", "Vendor:", "VRAM")):
        device_lines.append(stripped)

report = {
    "schema": "navatala_gpu.metal_validation.v1",
    "timingMode": "ctest_runtime_profile",
    "generatedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "commit": commit,
    "host": {
        "os": platform.system(),
        "arch": platform.machine(),
        "macOS": run_text(["/usr/bin/sw_vers", "-productVersion"]),
        "xcode": first_line(run_text(["/usr/bin/xcodebuild", "-version"])),
        "deviceSummary": "; ".join(device_lines),
    },
    "build": {
        "buildDir": str(build_dir),
        "cmakeVersion": first_line(run_text(["cmake", "--version"])),
        "compiler": first_line(run_text([os.environ.get("CXX", "c++"), "--version"])),
        "configuration": "Release",
    },
    "tests": {
        "regex": test_regex,
    },
    "runs": runs,
}

result_json.parent.mkdir(parents=True, exist_ok=True)
result_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
PY

python3 "$ROOT_DIR/scripts/validate_metal_validation_json.py" "$RESULT_JSON" --require-full
python3 "$ROOT_DIR/scripts/render_metal_validation_report.py" "$RESULT_JSON" --require-full --output "$RESULT_REPORT"

echo "Wrote:"
echo "  $RESULT_JSON"
echo "  $RESULT_REPORT"
echo "  $RESULT_DIR/baseline.log"
echo "  $RESULT_DIR/batched.log"
if [[ "$RUN_PRIVATE" == "1" ]]; then
  echo "  $RESULT_DIR/private.log"
fi
