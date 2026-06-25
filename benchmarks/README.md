# ROCm Vendor Benchmarks

This directory contains optional ROCm benchmarks used to answer AMD reviewer
feedback. They compare selected Navatala HIP kernels against ROCm vendor
libraries on real AMD hardware.

Metal runtime validation artifacts use the same `benchmarks/results/` and
`benchmarks/fixtures/` convention, but are documented separately in
[`docs/benchmarks/METAL_VALIDATION.md`](../docs/benchmarks/METAL_VALIDATION.md).

The benchmark is intentionally not part of the default build. It requires:

- ROCm with HIP compiler/runtime;
- rocBLAS;
- rocSPARSE;
- optionally hipSPARSELt, which is detected and reported when installed.

## Build And Run

From the public `navatala_gpu` tree:

```bash
./scripts/run_rocm_vendor_benchmarks.sh
```

The script configures a HIP-only benchmark build:

```bash
cmake -S . -B build-rocm-bench \
  -DNAVATALA_GPU_USE_HIP=ON \
  -DNAVATALA_GPU_USE_CUDA=OFF \
  -DNAVATALA_GPU_USE_VULKAN=OFF \
  -DNAVATALA_GPU_USE_OPENCL=OFF \
  -DNAVATALA_GPU_USE_METAL=OFF \
  -DNAVATALA_GPU_BUILD_BENCHMARKS=ON \
  -DNAVATALA_GPU_BUILD_EXAMPLES=OFF \
  -DNAVATALA_GPU_BUILD_TESTS=OFF \
  -DNAVATALA_GPU_BUILD_BENCHMARK_TESTS=ON
cmake --build build-rocm-bench -j
./build-rocm-bench/benchmarks/rocm_vendor_benchmark \
  --matrix standard \
  --iterations 50 \
  --warmup 10 \
  --json benchmarks/results/rocm_vendor_benchmark.json
```

After building, the benchmark binary can list its covered operations without
launching any kernels:

```bash
./build-rocm-bench/benchmarks/rocm_vendor_benchmark --list
```

Use `--matrix broad` or `NAVATALA_GPU_ROCM_BENCH_MATRIX=broad` to sweep
multiple sizes for each covered operation. Use `--matrix cta128-evidence` for
the CTA128 hardening matrix: it records the portable F16/F32-output
denominator, CTA64 shared, CTA64 pipelined, and CTA128 rows for 128³, 512³,
1024³, and representative rectangular tile-divisible shapes. Use
`--matrix edge-tails` for the focused Phase 1 tail matrix: it records true
non-divisible M/N/K shapes through direct CTA64_EDGE/CTA128_EDGE rows and
public-wrapper `MFMA`, `AUTO`, and `VENDOR` policy rows. Use
`--matrix wrapper-semantics --include-wrapper-gemm-benchmark` for a focused
public-ABI run covering small and large tail shapes, alpha/beta, transpose, and
strided batching without running the full broad matrix. The runner enables the
MFMA benchmark flag automatically for this matrix so the report also includes
direct semantic full-tile comparator rows for alpha/beta and transpose. Quick
mode always uses a small smoke-only matrix set and ignores the matrix
selection.

Set `NAVATALA_GPU_ROCM_BENCH_INCLUDE_MFMA=1` or pass
`--include-mfma-benchmark` to add the experimental HIP/gfx942 MFMA F16/F32 GEMM
row. This row is built only when requested because it requires the generated
matrix-intrinsic HIP kernel and ROCm 6.2+ toolchain support.

Set `NAVATALA_GPU_ROCM_BENCH_INCLUDE_WRAPPER_GEMM=1` or pass
`--include-wrapper-gemm-benchmark` to add public C ABI wrapper rows. This
includes the existing F32 vendor-dispatch row and the F16-input/F32-output MFMA
wrapper row. The latter calls `navatala_gpu_gemm_f16_f32` with
`NAVATALA_GPU_GEMM_IMPL=mfma`, pins `NAVATALA_GPU_GEMM_VENDOR_MODE=auto`, uses
the runtime CTA64/CTA128 auto policy, and covers the edge-capable wrapper path
rather than the raw tile-divisible timing kernels.

The broad matrix emits `GEMM_F16_PORTABLE_F32OUT` at 128³, 512³, and 1024³.
These rows run the generated `navatala_transformer_tiled_gemm_f16_f32_out`
kernel and are the apples-to-apples denominator for MFMA speedup tracking:

```text
GEMM_F16_PORTABLE_F32OUT.generatedMeanMs / GEMM_F16_MFMA_CTA64_DIRECT.generatedMeanMs
GEMM_F16_PORTABLE_F32OUT.generatedMeanMs / GEMM_F16_MFMA_CTA64_SHARED.generatedMeanMs
GEMM_F16_PORTABLE_F32OUT.generatedMeanMs / GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER.generatedMeanMs
GEMM_F16_PORTABLE_F32OUT.generatedMeanMs / GEMM_F16_MFMA_CTA64_SHARED_PADDED.generatedMeanMs
GEMM_F16_PORTABLE_F32OUT.generatedMeanMs / GEMM_F16_MFMA_CTA64_PIPELINED.generatedMeanMs
GEMM_F16_PORTABLE_F32OUT.generatedMeanMs / GEMM_F16_MFMA_CTA128.generatedMeanMs
```

Only same-host reports containing both rows for the same shape can evaluate
that gate.

For the current MFMA tuning line, the runtime wrapper policy is shape-aware:
CTA64 shared remains the medium-shape candidate, while CTA128 is the large-shape
candidate. The public `navatala_gpu_gemm_f16_f32` wrapper dispatches to the
full-tile CTA64/CTA128 kernels for aligned `NN`, alpha=1, beta=0 calls; to
full-tile semantic CTA64/CTA128 kernels for tile-divisible single-batch
alpha/beta or transpose calls; and to the edge-capable HIP/gfx942 kernels for
tail tiles or batching. Tail/general semantic calls default to CTA64_EDGE while
CTA128_EDGE remains available through `NAVATALA_GPU_GEMM_MFMA_MODE=cta128` for
diagnostics. The raw benchmark rows remain tile-divisible prototype timing rows
so they can be compared cleanly against earlier fixtures. The broad wrapper
matrix also emits explicit `GEMM_F16_F32_WRAPPER_MFMA_ALPHA_BETA`,
`GEMM_F16_F32_WRAPPER_MFMA_TRANSPOSE`, and
`GEMM_F16_F32_WRAPPER_MFMA_BATCHED` rows for hardware validation of the general
public ABI semantics. It also emits `GEMM_F16_F32_WRAPPER_AUTO_*` and
`GEMM_F16_F32_WRAPPER_VENDOR_*` comparator rows for measured vendor-favorable
cases. The `AUTO` rows prove the default policy chooses the vendor path when it
should; the `VENDOR` rows prove the explicit escape hatch remains correct.

For the focused public-ABI semantic sweep:

```bash
./scripts/run_rocm_vendor_benchmarks.sh \
  --matrix wrapper-semantics \
  --include-wrapper-gemm-benchmark \
  --iterations 30 \
  --warmup 8
```

This matrix emits direct `GEMM_F16_MFMA_CTA64_SHARED_SEMANTIC_ALPHA_BETA` and
`GEMM_F16_MFMA_CTA64_SHARED_SEMANTIC_TRANSPOSE` rows before the corresponding
wrapper rows. These rows separate Phase 2 kernel cost from public-wrapper
routing overhead.

To capture current CTA128 resource metadata alongside a timing report, run:

```bash
./scripts/run_rocm_vendor_benchmarks.sh \
  --matrix cta128-evidence \
  --include-mfma-benchmark \
  --iterations 30 \
  --warmup 8 \
  --profile-cta128
```

The profile step writes `profile/mfma_resource_metadata.csv`; the CTA128 row
must remain `Scratch_Size=0` before it can advance as a production candidate.

For the focused true-tail matrix:

```bash
./scripts/run_rocm_vendor_benchmarks.sh \
  --matrix edge-tails \
  --iterations 30 \
  --warmup 8
```

The runner automatically enables MFMA and wrapper GEMM rows for this preset.
Use this when evaluating Phase 1 edge/tail performance before running the
broader matrix.

The CTA64 NN-only fast-tail candidate is intentionally narrower than the
generic EDGE and split rows. Benchmark it only on `NN`, `alpha=1`, `beta=0`,
single-batch true-tail cases, and keep public `AUTO` rows in the same fixture
so the release policy remains evidence-based. The
`20260625_mi300x_edge_nn_rocm724` fixture validates this candidate on MI300X
with ROCm 7.2.4. It remains correctness-clean and scratch-free, with lower
resource pressure than generic CTA64_EDGE (`VGPR=24`, `SGPR=32` versus
`VGPR=40`, `SGPR=48`), but wall-time improves by only about 2-4 percent on the
sampled true-tail shapes. Public `AUTO` therefore remains vendor-routed for
these large true-tail shapes.

To capture rocprof resource metadata for the same true-tail kernels:

```bash
./scripts/run_rocm_vendor_benchmarks.sh \
  --matrix edge-tails \
  --iterations 30 \
  --warmup 8 \
  --profile-edge-tails
```

The profile step writes
`profile_edge_tails/edge_tail_resource_metadata.csv` with one row per
CTA64_EDGE / CTA64_EDGE_NN / CTA128_EDGE kernel. As with the CTA128 profile,
the profile pass caps iterations and warmup internally to keep the rocprof run
short; use the top-level JSON for publication timings and the profile CSV for
resource metadata.

Use `--matrix cfd-fixture` to benchmark a persisted CSR pressure-matrix fixture:

```bash
./scripts/run_rocm_vendor_benchmarks.sh \
  --matrix cfd-fixture \
  --csr-fixture-json benchmarks/fixtures/cfd_pressure_matrix_case_YYYYMMDD.json \
  --csr-fixture-bin benchmarks/fixtures/cfd_pressure_matrix_case_YYYYMMDD.csr.bin \
  --iterations 50 \
  --warmup 10
```

The direct benchmark binary accepts the same fixture arguments:

```bash
./build-rocm-bench/benchmarks/rocm_vendor_benchmark \
  --matrix cfd-fixture \
  --csr-fixture-json benchmarks/fixtures/cfd_pressure_matrix_case_YYYYMMDD.json \
  --csr-fixture-bin benchmarks/fixtures/cfd_pressure_matrix_case_YYYYMMDD.csr.bin \
  --json benchmarks/results/rocm_vendor_benchmark_cfd_fixture.json
```

For post-tuning SpMV reports, the broad matrix should include rowNnz 7, 15,
and 27 cases. rowNnz=15 is the transition case used to tune the thread-per-row
versus subgroup-per-row threshold; rowNnz=27 approximates common unstructured
CFD pressure-matrix rows.

The CFD fixture mode records two rows for the same matrix: one using the
generated thread-per-row CSR kernel and one using the generated subgroup-per-row
CSR kernel. This is intentional. The current generated kernels are whole-matrix
paths; fixture mode reports real row-distribution statistics and threshold
classification without pretending that a per-row adaptive kernel is already
present.

The convenience script also validates the JSON and renders a Markdown report:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/results/rocm_vendor_benchmark.json
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/results/rocm_vendor_benchmark.json \
  --output benchmarks/results/rocm_vendor_benchmark.md
```

To exercise the public JSON/report contract on a CPU-only machine, run:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/rocm_vendor_benchmark.sample.json
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/rocm_vendor_benchmark.sample.json \
  --output /tmp/rocm_vendor_benchmark.sample.md
```

The fixtures directory also carries a dated MI300X reference run:

```text
benchmarks/fixtures/rocm_vendor_benchmark_mi300x_20260619.json
```

That file is the durable record for the 2026-06-19 broad ROCm benchmark run
used by the tuning tickets. It is validated in CTest with full-matrix and
hipSPARSELt requirements enabled. Use it as a historical reference, not as a
fresh benchmark claim for a new release.

Benchmark reports label the selected implementation path:
`portable_kernel`, `tuned_kernel`, or `vendor_library`. In particular,
`navatala_transformer_tiled_gemm_f32` is a portable fallback/reference kernel;
public GEMM wrapper paths may separately dispatch through rocBLAS/cuBLAS when
the generated `LibraryOp.BLAS_GEMM` route is selected.

## Current Coverage

| Operation | Generated path | Vendor baseline |
| --- | --- | --- |
| AXPY F32 | `navatala_sparse_axpy_f32` | `rocblas_saxpy` |
| GEMM F32 | `navatala_transformer_tiled_gemm_f32` | `rocblas_sgemm` |
| GEMM F16 portable F32 output | `navatala_transformer_tiled_gemm_f16_f32_out` | `rocblas_gemm_ex` with F16 input/F32 output/F32 accumulation |
| GEMM F16 MFMA historical | `navatala_transformer_tiled_gemm_f16_mfma_k_loop` | `rocblas_gemm_ex` with F16 inputs/F32 accumulation |
| GEMM F16 MFMA CTA64 direct | `navatala_transformer_tiled_gemm_f16_mfma_cta64_direct` | `rocblas_gemm_ex` with F16 inputs/F32 accumulation |
| GEMM F16 MFMA CTA64 shared | `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared` | `rocblas_gemm_ex` with F16 inputs/F32 accumulation |
| GEMM F16 MFMA CTA64 shared early barrier | `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier` | `rocblas_gemm_ex` with F16 inputs/F32 accumulation |
| GEMM F16 MFMA CTA64 shared padded | `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded` | `rocblas_gemm_ex` with F16 inputs/F32 accumulation |
| GEMM F16 MFMA CTA64 pipelined | `navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined` | `rocblas_gemm_ex` with F16 inputs/F32 accumulation |
| GEMM F16 MFMA CTA128 | `navatala_transformer_tiled_gemm_f16_mfma_cta128` | `rocblas_gemm_ex` with F16 inputs/F32 accumulation |
| GEMM F16/F32 wrapper MFMA | `navatala_gpu_gemm_f16_f32` with `NAVATALA_GPU_GEMM_IMPL=mfma` | `rocblas_gemm_ex` with F16 input/F32 output/F32 accumulation |
| CSR SpMV F32 | adaptive `navatala_graph_spmv_weighted_f32` / `navatala_graph_spmv_weighted_subgroup_f32` | `rocsparse_spmv` |
| Structured sparse GEMM F16 | `SparseLt_StructuredMatmul` benchmark path | `rocblas_hgemm` pruned-A dense reference |

hipSPARSELt is detected by CMake and reported in the JSON output. Set
`NAVATALA_GPU_ROCM_BENCH_INCLUDE_HIPSPARSELT=1` or pass
`--include-hipsparselt-benchmark` to add the structured sparse GEMM row. On
gfx942 this row uses FP16 input/output with FP32 compute, because float32
structured sparse matmul is not a supported hipSPARSELt path on MI300-class
hardware.

The MFMA rows are tuned-kernel rows, not production GEMM wrappers. The
historical `GEMM_F16_MFMA` row computes one 32x32 output tile per wave with
K-loop accumulation. The `GEMM_F16_MFMA_CTA64_DIRECT` row is a low-pressure
diagnostic variant: 64x64 CTA, four wave64s per workgroup, one accumulator per
wave, no shared-memory staging, tile-divisible NN only, alpha=1, beta=0. The
`GEMM_F16_MFMA_CTA64_SHARED` row keeps the CTA64 one-accumulator-per-wave
register geometry and stages a small 2 KB LDS panel per K step; it is the
current best measured tuned MFMA prototype. The
`GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER` and
`GEMM_F16_MFMA_CTA64_SHARED_PADDED` rows are diagnostics: the former preserves
the clean CTA64 resource envelope but was neutral on MI300X; the latter is
correct and spill-free but slower because it increases LDS/register pressure.
The `GEMM_F16_MFMA_CTA64_PIPELINED` row is the R6 two-slot staged-panel
contract row. Today it is expected to behave like a synchronous vector-staged
prototype because the admitted HIP lowerer has not yet enabled true async
overlap; it is included so same-host runs can validate correctness, resource
usage, and the impact of the formal R6 copy-group/panel path before async
lowering is admitted.
The `GEMM_F16_MFMA_CTA128` row is the larger shared-memory Phase-0
production-hardening prototype: 128x128x32 CTA, shared-memory staging, four
wave64s per workgroup, tile-divisible NN only, alpha=1, beta=0. Use the CTA64
direct/shared diagnostic rows and CTA128 row together for MI300/gfx942 speedup
and register-pressure diagnosis against `GEMM_F16_PORTABLE_F32OUT`.

The focused `edge-tails` matrix also emits
`GEMM_F16_MFMA_CTA64_SHARED_EDGE_NN`. This is a narrow true-tail candidate for
the common `NN`, `alpha=1`, `beta=0`, single-batch path. It removes transpose,
batch, and scaled-store branches from the generic CTA64 EDGE row so same-host
MI300X runs can measure whether reducing instruction count is enough before we
promote any generated EDGE route over vendor AUTO dispatch.

Wrapper validation can force high-level dispatch with
`NAVATALA_GPU_GEMM_IMPL=auto|vendor|mfma|portable`. `vendor` forces BLAS for
supported Float32 and mixed F16-input/F32-output public wrappers, `portable`
forces the fallback/reference path, and contradictory combinations with
`NAVATALA_GPU_GEMM_VENDOR_MODE` fail loudly. `mfma` selects the generated
HIP/gfx942 F16-input/F32-output wrapper (`navatala_gpu_gemm_f16_f32`) when the
HIP transformer registry shard is built into the runtime; the current Float32
ABI returns `NAVATALA_NOT_IMPLEMENTED` for that selector.

The opt-in `GEMM_F16_F32_WRAPPER_MFMA` row is the wrapper-level timing and
correctness row for that public ABI. It differs from the raw MFMA rows above:
it dispatches through the runtime registry, uses the edge-capable CTA64/CTA128
kernels, and its measured time includes wrapper scalar-parameter setup and the
necessary synchronization points. Treat it as user-facing wrapper evidence, not
as a replacement for the raw kernel micro-benchmark rows.

The `GEMM_F16_F32_WRAPPER_AUTO_*` rows run the default public ABI policy for
transpose, large-tail, and strided-batched cases where the current generated
MFMA path is correct but not yet the fastest measured implementation. The
matching `GEMM_F16_F32_WRAPPER_VENDOR_*` rows force the same public ABI through
rocBLAS to validate the explicit vendor escape hatch.

The `GEMM_F16_PORTABLE_F32OUT` row is a portable-kernel denominator, not a
tuned path. It is present so reports can evaluate MFMA speedup without relying
on either the unrelated F32 portable GEMM row or the legacy F16-output row.

## Output

The benchmark writes a JSON report with:

- operation and shape;
- matrix set (`quick`, `standard`, `broad`, `cta128-evidence`,
  `wrapper-semantics`, or `cfd-fixture`);
- Navatala HIP kernel path;
- vendor library baseline;
- correctness result and max absolute error;
- `timingMode: back_to_back_throughput_mean_per_launch`;
- generated/vendor mean milliseconds per launch from HIP events;
- generated/vendor runtime ratio;
- hipSPARSELt availability and mode.

Wrapper rows may also include `hostSubmitMeanMs`, `hostWallMeanMs`,
`hostSubmitOverGeneratedRatio`, and `hostWallOverGeneratedRatio`. These fields
are diagnostics for wrapper-overhead investigations: submit time is CPU wall
time to invoke the public wrapper and enqueue work; wall time also includes the
final event wait. Older fixtures do not contain these optional fields.

For `--matrix cfd-fixture`, the result notes include row-distribution metadata:
`min/mean/p50/p90/p99/max`, the active
`NAVATALA_GPU_SPMV_SUBGROUP_THRESHOLD`, row-length deciles
`0,10,...,100`, the threshold-selected path for each decile, and how many rows
fall below or above the threshold.

## CSR Fixture Binary Layout

The fixture binary is native little-endian and versioned. It is deliberately
simple so OpenFOAM GPU exporters and post-processing scripts can write it
without linking against the benchmark harness.

Header:

| Field | Type | Notes |
| --- | --- | --- |
| magic | `char[16]` | Bytes `NVCSRSPMVF32\0\0\0\0`. |
| version | `uint32` | Must be `1`. |
| indexBase | `uint32` | Must be `0` for the current benchmark. |
| flags | `uint32` | Reserved, write `0`. |
| reserved | `uint32` | Reserved, write `0`. |
| rows | `uint64` | CSR row count. |
| cols | `uint64` | CSR column count. |
| nnz | `uint64` | Number of non-zeros. |

Payload immediately follows the header:

| Field | Type | Count |
| --- | --- | ---: |
| rowPtr | `uint32` | `rows + 1` |
| colIdx | `uint32` | `nnz` |
| values | `float32` | `nnz` |
| x | `float32` | `cols` |

The JSON sidecar is not parsed by the benchmark for core matrix data, but is
carried into the result notes and should include enough provenance to reproduce
the matrix:

- source case name and git commit;
- rows, cols, nnz;
- min/mean/p50/p90/p99/max row nnz;
- index base;
- dtype;
- whether diagonal and boundary rows are included;
- binary layout string, for example `navatala_gpu.csr_spmv_f32.v1`.

The wrapper writes three artifacts:

- `benchmarks/results/rocm_vendor_benchmark.json`
- `benchmarks/results/rocm_vendor_benchmark.log`
- `benchmarks/results/rocm_vendor_benchmark.md`

Set `NAVATALA_GPU_ROCM_BENCH_REQUIRE_FULL=1` to reject quick-mode reports in
the post-run validator. Set `NAVATALA_GPU_ROCM_BENCH_REQUIRE_HIPSPARSELT=1` to
fail a run where hipSPARSELt was not detected. Set
`NAVATALA_GPU_ROCM_BENCH_INCLUDE_HIPSPARSELT=1` to include the opt-in
hipSPARSELt benchmark row; the JSON then reports
`hipSPARSELtMode: "vendor_benchmark"`.

Use `docs/benchmarks/ROCM_VALIDATION_TEMPLATE.md` to turn the JSON and CI logs
into a public release validation report.

For runtime correctness smoke tests, run:

```bash
./scripts/run_rocm_runtime_smoke.sh
```

That script builds a HIP-only runtime/test configuration and runs the core CTest
set with `GPU_RUNTIME_BACKEND=hip`.
