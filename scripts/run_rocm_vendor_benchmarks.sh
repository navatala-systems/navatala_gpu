#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
#
# Build and run the ROCm vendor comparison benchmark.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${NAVATALA_GPU_ROCM_BENCH_BUILD_DIR:-$ROOT_DIR/build-rocm-bench}"
RESULT_DIR="${NAVATALA_GPU_ROCM_BENCH_RESULT_DIR:-$ROOT_DIR/benchmarks/results}"
ROCM_ROOT="${NAVATALA_GPU_ROCM_ROOT:-${ROCM_PATH:-${HIP_PATH:-}}}"
ITERATIONS="${NAVATALA_GPU_ROCM_BENCH_ITERATIONS:-50}"
WARMUP="${NAVATALA_GPU_ROCM_BENCH_WARMUP:-10}"
QUICK="${NAVATALA_GPU_ROCM_BENCH_QUICK:-0}"
MATRIX="${NAVATALA_GPU_ROCM_BENCH_MATRIX:-standard}"
PROFILE_CTA128="${NAVATALA_GPU_ROCM_BENCH_PROFILE_CTA128:-0}"
PROFILE_EDGE_TAILS="${NAVATALA_GPU_ROCM_BENCH_PROFILE_EDGE_TAILS:-0}"
CSR_FIXTURE_JSON="${NAVATALA_GPU_ROCM_BENCH_CSR_FIXTURE_JSON:-}"
CSR_FIXTURE_BIN="${NAVATALA_GPU_ROCM_BENCH_CSR_FIXTURE_BIN:-}"
REQUIRE_FULL="${NAVATALA_GPU_ROCM_BENCH_REQUIRE_FULL:-0}"
REQUIRE_HIPSPARSELT="${NAVATALA_GPU_ROCM_BENCH_REQUIRE_HIPSPARSELT:-0}"
INCLUDE_HIPSPARSELT="${NAVATALA_GPU_ROCM_BENCH_INCLUDE_HIPSPARSELT:-0}"
INCLUDE_WRAPPER_GEMM="${NAVATALA_GPU_ROCM_BENCH_INCLUDE_WRAPPER_GEMM:-0}"
INCLUDE_MFMA="${NAVATALA_GPU_ROCM_BENCH_INCLUDE_MFMA:-0}"
RESULT_JSON="$RESULT_DIR/rocm_vendor_benchmark.json"
RESULT_REPORT="$RESULT_DIR/rocm_vendor_benchmark.md"

usage() {
  cat <<'EOF'
Usage: run_rocm_vendor_benchmarks.sh [options]

Options:
  --quick                         Run small smoke shapes instead of the selected matrix.
  --iterations N                  Benchmark iterations.
  --warmup N                      Warmup iterations.
  --matrix NAME                   Matrix preset: standard, broad, cta128-evidence, edge-tails, wrapper-semantics, cfd-fixture.
  --profile-cta128                Capture rocprof resource metadata for CTA128 MFMA rows.
  --profile-edge-tails            Capture rocprof resource metadata for CTA64/CTA128 EDGE rows on true-tail shapes.
  --csr-fixture-json PATH         CSR fixture JSON sidecar for cfd-fixture mode.
  --csr-fixture-bin PATH          CSR fixture binary for cfd-fixture mode.
  --json PATH                     Output JSON path.
  --build-dir PATH                CMake build directory.
  --result-dir PATH               Result directory for log/report/default JSON.
  --require-full                  Require all full benchmark rows in validation.
  --require-hipsparselt           Require hipSPARSELt availability/row in validation.
  --include-hipsparselt-benchmark Include hipSPARSELt benchmark row when available.
  --include-wrapper-gemm-benchmark Include runtime-wrapper GEMM benchmark row.
  --include-mfma-benchmark        Include HIP/gfx942 MFMA F16/F32 benchmark row.
  --rocm-root PATH                ROCm installation root, e.g. /opt/rocm-7.2.4.
  -h, --help                      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
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
    --matrix)
      MATRIX="${2:?missing value for --matrix}"
      shift 2
      ;;
    --matrix=*)
      MATRIX="${1#*=}"
      shift
      ;;
    --profile-cta128)
      PROFILE_CTA128=1
      shift
      ;;
    --profile-edge-tails)
      PROFILE_EDGE_TAILS=1
      shift
      ;;
    --csr-fixture-json)
      CSR_FIXTURE_JSON="${2:?missing value for --csr-fixture-json}"
      shift 2
      ;;
    --csr-fixture-json=*)
      CSR_FIXTURE_JSON="${1#*=}"
      shift
      ;;
    --csr-fixture-bin)
      CSR_FIXTURE_BIN="${2:?missing value for --csr-fixture-bin}"
      shift 2
      ;;
    --csr-fixture-bin=*)
      CSR_FIXTURE_BIN="${1#*=}"
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
      RESULT_JSON="$RESULT_DIR/rocm_vendor_benchmark.json"
      RESULT_REPORT="$RESULT_DIR/rocm_vendor_benchmark.md"
      shift 2
      ;;
    --result-dir=*)
      RESULT_DIR="${1#*=}"
      RESULT_JSON="$RESULT_DIR/rocm_vendor_benchmark.json"
      RESULT_REPORT="$RESULT_DIR/rocm_vendor_benchmark.md"
      shift
      ;;
    --require-full)
      REQUIRE_FULL=1
      shift
      ;;
    --require-hipsparselt)
      REQUIRE_HIPSPARSELT=1
      shift
      ;;
    --include-hipsparselt-benchmark)
      INCLUDE_HIPSPARSELT=1
      shift
      ;;
    --include-wrapper-gemm-benchmark)
      INCLUDE_WRAPPER_GEMM=1
      shift
      ;;
    --include-mfma-benchmark)
      INCLUDE_MFMA=1
      shift
      ;;
    --rocm-root)
      ROCM_ROOT="${2:?missing value for --rocm-root}"
      shift 2
      ;;
    --rocm-root=*)
      ROCM_ROOT="${1#*=}"
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
mkdir -p "$(dirname "$RESULT_JSON")"

if [[ -z "$ROCM_ROOT" ]]; then
  if [[ -d /opt/rocm ]]; then
    ROCM_ROOT="/opt/rocm"
  else
    # Some cloud images install versioned ROCm roots without an /opt/rocm
    # symlink, for example /opt/rocm-7.2.4.
    ROCM_ROOT="$(find /opt -maxdepth 1 -type d -name 'rocm*' 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
fi

if [[ -n "$ROCM_ROOT" && -d "$ROCM_ROOT" ]]; then
  export ROCM_PATH="$ROCM_ROOT"
  export HIP_PATH="$ROCM_ROOT"
  export PATH="$ROCM_ROOT/bin:$PATH"
fi

if [[ "$QUICK" != "1" && "$MATRIX" == "wrapper-semantics" ]]; then
  # The focused semantic matrix includes direct MFMA comparator rows alongside
  # public-wrapper rows, so build and pass the MFMA benchmark flag implicitly.
  INCLUDE_MFMA=1
fi
if [[ "$QUICK" != "1" && "$MATRIX" == "edge-tails" ]]; then
  # The focused tail matrix exists to compare direct EDGE kernels with public
  # wrapper policy rows, so enable both sides implicitly.
  INCLUDE_MFMA=1
  INCLUDE_WRAPPER_GEMM=1
fi
if [[ "$PROFILE_EDGE_TAILS" == "1" ]]; then
  INCLUDE_MFMA=1
  INCLUDE_WRAPPER_GEMM=1
fi

HIP_CMAKE_COMPILER=""
if [[ -n "$ROCM_ROOT" && -x "$ROCM_ROOT/bin/amdclang++" ]]; then
  HIP_CMAKE_COMPILER="$ROCM_ROOT/bin/amdclang++"
elif [[ -n "$ROCM_ROOT" && -x "$ROCM_ROOT/lib/llvm/bin/clang++" ]]; then
  HIP_CMAKE_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang++"
fi

echo "=== ROCm environment ==="
if [[ -n "$ROCM_ROOT" ]]; then
  echo "ROCm root: $ROCM_ROOT"
fi
command -v rocminfo >/dev/null 2>&1 && rocminfo | sed -n '1,80p' || true
command -v hipcc >/dev/null 2>&1 && hipcc --version || true
command -v rocm-smi >/dev/null 2>&1 && rocm-smi || true
if command -v rocminfo >/dev/null 2>&1; then
  export NAVATALA_GPU_ROCMINFO_SUMMARY="$(
    rocminfo \
      | awk '/Name:|Marketing Name:|Agent [0-9]+|ISA:/ { gsub(/^[[:space:]]+/, ""); print }' \
      | sed -n '1,40p' \
      | paste -sd '; ' -
  )"
else
  export NAVATALA_GPU_ROCMINFO_SUMMARY=""
fi

MFMA_CMAKE=OFF
if [[ "$INCLUDE_MFMA" == "1" ]]; then
  MFMA_CMAKE=ON
fi

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
  -DNAVATALA_GPU_BUILD_TESTS=OFF \
  -DNAVATALA_GPU_BUILD_BENCHMARKS=ON \
  -DNAVATALA_GPU_BUILD_BENCHMARK_TESTS=ON \
  -DNAVATALA_GPU_BUILD_MFMA_BENCHMARK="$MFMA_CMAKE" \
  -DGPU_RUNTIME_NAVATALA_FFI_STUB=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" -j"$(nproc)"

bench_args=(
  --iterations "$ITERATIONS"
  --warmup "$WARMUP"
  --json "$RESULT_JSON"
)
if [[ "$QUICK" == "1" ]]; then
  bench_args+=(--quick)
else
  bench_args+=(--matrix "$MATRIX")
fi
if [[ -n "$CSR_FIXTURE_JSON" ]]; then
  bench_args+=(--csr-fixture-json "$CSR_FIXTURE_JSON")
fi
if [[ -n "$CSR_FIXTURE_BIN" ]]; then
  bench_args+=(--csr-fixture-bin "$CSR_FIXTURE_BIN")
fi
if [[ "$REQUIRE_HIPSPARSELT" == "1" ]]; then
  bench_args+=(--require-hipsparselt)
fi
if [[ "$INCLUDE_HIPSPARSELT" == "1" ]]; then
  bench_args+=(--include-hipsparselt-benchmark)
fi
if [[ "$INCLUDE_WRAPPER_GEMM" == "1" ]]; then
  bench_args+=(--include-wrapper-gemm-benchmark)
fi
if [[ "$INCLUDE_MFMA" == "1" ]]; then
  bench_args+=(--include-mfma-benchmark)
fi

"$BUILD_DIR/benchmarks/rocm_vendor_benchmark" "${bench_args[@]}" \
  | tee "$RESULT_DIR/rocm_vendor_benchmark.log"

validator_args=("$RESULT_JSON")
report_args=("$RESULT_JSON" --output "$RESULT_REPORT")
if [[ "$REQUIRE_FULL" == "1" ]]; then
  validator_args+=(--require-full)
  report_args+=(--require-full)
fi
if [[ "$REQUIRE_HIPSPARSELT" == "1" ]]; then
  validator_args+=(--require-hipsparselt)
  report_args+=(--require-hipsparselt)
fi

python3 "$ROOT_DIR/scripts/validate_rocm_benchmark_json.py" "${validator_args[@]}"
python3 "$ROOT_DIR/scripts/render_rocm_benchmark_report.py" "${report_args[@]}"

echo "Wrote:"
echo "  $RESULT_JSON"
echo "  $RESULT_DIR/rocm_vendor_benchmark.log"
echo "  $RESULT_REPORT"

if [[ "$PROFILE_CTA128" == "1" ]]; then
  if ! command -v rocprofv3 >/dev/null 2>&1; then
    echo "ERROR: --profile-cta128 requires rocprofv3 on PATH" >&2
    exit 2
  fi
  if [[ "$INCLUDE_MFMA" != "1" ]]; then
    echo "ERROR: --profile-cta128 requires --include-mfma-benchmark" >&2
    exit 2
  fi

  PROFILE_DIR="$RESULT_DIR/profile"
  PROFILE_JSON="$PROFILE_DIR/rocm_vendor_benchmark.json"
  mkdir -p "$PROFILE_DIR"
  rm -f "$PROFILE_DIR"/rocprof_cta128_*.csv "$PROFILE_JSON" "$PROFILE_DIR/mfma_resource_metadata.csv"

  profile_iterations="$ITERATIONS"
  if (( profile_iterations > 5 )); then
    profile_iterations=5
  fi
  profile_warmup="$WARMUP"
  if (( profile_warmup > 1 )); then
    profile_warmup=1
  fi

  echo "=== ROCm CTA128 profile ==="
  rocprofv3 \
    --kernel-trace \
    --scratch-memory-trace \
    --stats \
    --output-format csv \
    --output-directory "$PROFILE_DIR" \
    --output-file rocprof_cta128 \
    --kernel-include-regex "navatala_transformer_tiled_gemm_f16_mfma_cta128" \
    -- "$BUILD_DIR/benchmarks/rocm_vendor_benchmark" \
      --matrix cta128-evidence \
      --iterations "$profile_iterations" \
      --warmup "$profile_warmup" \
      --include-mfma-benchmark \
      --json "$PROFILE_JSON" \
    | tee "$PROFILE_DIR/rocm_vendor_benchmark.log"

  python3 "$ROOT_DIR/scripts/validate_rocm_benchmark_json.py" "$PROFILE_JSON"
  python3 "$ROOT_DIR/scripts/render_rocm_benchmark_report.py" \
    "$PROFILE_JSON" \
    --output "$PROFILE_DIR/rocm_vendor_benchmark.md"
  python3 "$ROOT_DIR/scripts/summarize_rocm_kernel_resources.py" \
    "$PROFILE_DIR" \
    --kernel-substring navatala_transformer_tiled_gemm_f16_mfma_cta128 \
    --output "$PROFILE_DIR/mfma_resource_metadata.csv"

  echo "Wrote CTA128 profile:"
  echo "  $PROFILE_JSON"
  echo "  $PROFILE_DIR/rocm_vendor_benchmark.log"
  echo "  $PROFILE_DIR/rocm_vendor_benchmark.md"
  echo "  $PROFILE_DIR/mfma_resource_metadata.csv"
fi

if [[ "$PROFILE_EDGE_TAILS" == "1" ]]; then
  if ! command -v rocprofv3 >/dev/null 2>&1; then
    echo "ERROR: --profile-edge-tails requires rocprofv3 on PATH" >&2
    exit 2
  fi
  if [[ "$INCLUDE_MFMA" != "1" ]]; then
    echo "ERROR: --profile-edge-tails requires --include-mfma-benchmark" >&2
    exit 2
  fi

  EDGE_PROFILE_DIR="$RESULT_DIR/profile_edge_tails"
  EDGE_PROFILE_JSON="$EDGE_PROFILE_DIR/rocm_vendor_benchmark.json"
  mkdir -p "$EDGE_PROFILE_DIR"
  rm -f "$EDGE_PROFILE_DIR"/rocprof_edge_tails_*.csv \
        "$EDGE_PROFILE_JSON" \
        "$EDGE_PROFILE_DIR/edge_tail_resource_metadata.csv"

  edge_profile_iterations="$ITERATIONS"
  if (( edge_profile_iterations > 5 )); then
    edge_profile_iterations=5
  fi
  edge_profile_warmup="$WARMUP"
  if (( edge_profile_warmup > 1 )); then
    edge_profile_warmup=1
  fi

  echo "=== ROCm EDGE tail profile ==="
  rocprofv3 \
    --kernel-trace \
    --scratch-memory-trace \
    --stats \
    --output-format csv \
    --output-directory "$EDGE_PROFILE_DIR" \
    --output-file rocprof_edge_tails \
    --kernel-include-regex "navatala_transformer_tiled_gemm_f16_mfma_cta.*edge" \
    -- "$BUILD_DIR/benchmarks/rocm_vendor_benchmark" \
      --matrix edge-tails \
      --iterations "$edge_profile_iterations" \
      --warmup "$edge_profile_warmup" \
      --include-mfma-benchmark \
      --include-wrapper-gemm-benchmark \
      --json "$EDGE_PROFILE_JSON" \
    | tee "$EDGE_PROFILE_DIR/rocm_vendor_benchmark.log"

  python3 "$ROOT_DIR/scripts/validate_rocm_benchmark_json.py" "$EDGE_PROFILE_JSON"
  python3 "$ROOT_DIR/scripts/render_rocm_benchmark_report.py" \
    "$EDGE_PROFILE_JSON" \
    --output "$EDGE_PROFILE_DIR/rocm_vendor_benchmark.md"
  python3 "$ROOT_DIR/scripts/summarize_rocm_kernel_resources.py" \
    "$EDGE_PROFILE_DIR" \
    --output "$EDGE_PROFILE_DIR/edge_tail_resource_metadata.csv"

  echo "Wrote EDGE tail profile:"
  echo "  $EDGE_PROFILE_JSON"
  echo "  $EDGE_PROFILE_DIR/rocm_vendor_benchmark.log"
  echo "  $EDGE_PROFILE_DIR/rocm_vendor_benchmark.md"
  echo "  $EDGE_PROFILE_DIR/edge_tail_resource_metadata.csv"
fi
