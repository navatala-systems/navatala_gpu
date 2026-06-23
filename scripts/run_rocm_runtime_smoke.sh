#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
#
# Build and run the HIP/ROCm runtime smoke tests on an AMD GPU host.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${NAVATALA_GPU_ROCM_SMOKE_BUILD_DIR:-$ROOT_DIR/build-rocm-smoke}"
RESULT_DIR="${NAVATALA_GPU_ROCM_SMOKE_RESULT_DIR:-$ROOT_DIR/benchmarks/results}"
TEST_REGEX="${NAVATALA_GPU_ROCM_SMOKE_REGEX:-test_device_selection|test_events|test_graphs|test_kernel_execution|test_program_cache|manifest_conformance|rocm_benchmark_json_fixture_conformance|rocm_benchmark_report_fixture_render}"
ROCM_ROOT="${NAVATALA_GPU_ROCM_ROOT:-${ROCM_PATH:-${HIP_PATH:-}}}"

usage() {
  cat <<'EOF'
Usage: run_rocm_runtime_smoke.sh [options]

Options:
  --rocm-root PATH    ROCm installation root, e.g. /opt/rocm-7.2.4.
  --build-dir PATH    CMake build directory.
  --result-dir PATH   Result directory for the smoke log.
  --test-regex REGEX  CTest regular expression.
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rocm-root)
      ROCM_ROOT="${2:?missing value for --rocm-root}"
      shift 2
      ;;
    --rocm-root=*)
      ROCM_ROOT="${1#*=}"
      shift
      ;;
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
      shift 2
      ;;
    --result-dir=*)
      RESULT_DIR="${1#*=}"
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

if [[ -z "$ROCM_ROOT" ]]; then
  if [[ -d /opt/rocm ]]; then
    ROCM_ROOT="/opt/rocm"
  else
    ROCM_ROOT="$(find /opt -maxdepth 1 -type d -name 'rocm*' 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
fi

if [[ -n "$ROCM_ROOT" && -d "$ROCM_ROOT" ]]; then
  export ROCM_PATH="$ROCM_ROOT"
  export HIP_PATH="$ROCM_ROOT"
  export PATH="$ROCM_ROOT/bin:$PATH"
fi

HIP_CMAKE_COMPILER=""
if [[ -n "$ROCM_ROOT" && -x "$ROCM_ROOT/bin/amdclang++" ]]; then
  HIP_CMAKE_COMPILER="$ROCM_ROOT/bin/amdclang++"
elif [[ -n "$ROCM_ROOT" && -x "$ROCM_ROOT/lib/llvm/bin/clang++" ]]; then
  HIP_CMAKE_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang++"
fi

echo "=== ROCm runtime smoke environment ==="
if [[ -n "$ROCM_ROOT" ]]; then
  echo "ROCm root: $ROCM_ROOT"
fi
command -v rocminfo >/dev/null 2>&1 && rocminfo | sed -n '1,80p' || true
command -v hipcc >/dev/null 2>&1 && hipcc --version || true
command -v rocm-smi >/dev/null 2>&1 && rocm-smi || true

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  ${ROCM_ROOT:+-DCMAKE_PREFIX_PATH="$ROCM_ROOT"} \
  ${HIP_CMAKE_COMPILER:+-DCMAKE_HIP_COMPILER="$HIP_CMAKE_COMPILER"} \
  ${ROCM_ROOT:+-DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_ROOT"} \
  -DNAVATALA_GPU_USE_HIP=ON \
  -DNAVATALA_GPU_USE_CUDA=OFF \
  -DNAVATALA_GPU_USE_VULKAN=OFF \
  -DNAVATALA_GPU_USE_OPENCL=OFF \
  -DNAVATALA_GPU_USE_METAL=OFF \
  -DNAVATALA_GPU_BUILD_RUNTIME=ON \
  -DNAVATALA_GPU_BUILD_EXAMPLES=OFF \
  -DNAVATALA_GPU_BUILD_TESTS=ON \
  -DNAVATALA_GPU_BUILD_BENCHMARKS=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" -j"$(nproc)"

GPU_RUNTIME_BACKEND=hip \
ctest --test-dir "$BUILD_DIR" --output-on-failure -R "$TEST_REGEX" \
  | tee "$RESULT_DIR/rocm_runtime_smoke.log"

echo "Wrote:"
echo "  $RESULT_DIR/rocm_runtime_smoke.log"
