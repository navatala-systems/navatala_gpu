# ROCm Vendor Benchmark Harness

This document records the public ROCm benchmark harness added in response to
AMD reviewer feedback. The goal is to compare selected Navatala HIP kernels
against ROCm vendor-library baselines on real AMD hardware, without implying
that every HIP kernel is vendor-backed or hand-tuned.

## Scope

The harness lives in the public tree under:

- `benchmarks/rocm_vendor_benchmark.hip`
- `benchmarks/CMakeLists.txt`
- `scripts/run_rocm_runtime_smoke.sh`
- `scripts/run_rocm_vendor_benchmarks.sh`
- `.github/workflows/rocm-validation.yml`

It is opt-in and is not part of the default build. Enable it with:

```bash
cmake -S . -B build-rocm-bench \
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
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm-bench -j
./build-rocm-bench/benchmarks/rocm_vendor_benchmark \
  --matrix standard \
  --iterations 50 \
  --warmup 10 \
  --json benchmarks/results/rocm_vendor_benchmark.json
```

The convenience wrapper is:

```bash
./scripts/run_rocm_vendor_benchmarks.sh
```

After the benchmark target is built, list the covered operations without
launching GPU work:

```bash
./build-rocm-bench/benchmarks/rocm_vendor_benchmark --list
```

The wrapper performs three steps after the benchmark binary exits:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/results/rocm_vendor_benchmark.json
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/results/rocm_vendor_benchmark.json \
  --output benchmarks/results/rocm_vendor_benchmark.md
```

For local no-GPU checks of the artifact contract, use the fixture:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/rocm_vendor_benchmark.sample.json
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/rocm_vendor_benchmark.sample.json \
  --output /tmp/rocm_vendor_benchmark.sample.md
```

The repository also includes a current dated MI300X reference fixture:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/rocm_vendor_benchmark_mi300x_20260622_rocm724.json \
  --require-full \
  --require-hipsparselt
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/rocm_vendor_benchmark_mi300x_20260622_rocm724.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_20260622_rocm724.md
```

This reference fixture records the broad ROCm benchmark run from 2026-06-22 on
AMD Instinct MI300X VF hardware with ROCm 7.2.4. It includes wrapper GEMM vendor
dispatch rows, the rowNnz=15 and rowNnz=27 SpMV tuning rows, hipSPARSELt
structured-sparse rows, `rocminfo` provenance, and `kernelClass` metadata. It is
kept so tuning tickets and review notes can cite durable evidence rather than
temporary `/tmp` artifacts. Release claims should still use a fresh run.

The current public-wrapper semantic evidence is stored under:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_semantics/rocm_vendor_benchmark.json \
  --require-full
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_semantics/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_wrapper_semantics.md
```

This focused sweep validates the public F16-input/F32-output MFMA wrapper
against rocBLAS for tail tiles, alpha/beta, transpose, and strided batching:

| Row | Generated ms | Vendor ms | Ratio | Correctness |
| --- | ---: | ---: | ---: | --- |
| Tail small 513x511x257 | 0.023151 | 0.043337 | 0.534x | pass |
| Tail large 1023x1024x2003 | 0.185373 | 0.093138 | 1.990x | pass |
| Alpha/beta 512³ | 0.038582 | 0.033146 | 1.164x | pass |
| Transpose 384x512x256 | 0.025538 | 0.020408 | 1.251x | pass |
| Strided batch 256³ x3 | 0.018260 | 0.017542 | 1.041x | pass |

The strided-batch row confirms the benchmark builds and runs
`rocblas_gemm_strided_batched_ex` on the MI300X validation host.

The follow-up full-tile semantic evidence is stored under:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_semantic_fulltile_50iter/rocm_vendor_benchmark.json
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_semantic_fulltile_50iter/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_wrapper_semantic_fulltile_50iter.md
```

This rerun validates the full-tile semantic CTA64/CTA128 dispatch that avoids
the EDGE path for tile-divisible alpha/beta and transpose calls:

| Row | Generated ms | Vendor ms | Ratio | Correctness |
| --- | ---: | ---: | ---: | --- |
| Direct alpha/beta semantic 512³ | 0.025433 | 0.032781 | 0.776x | pass |
| Wrapper alpha/beta 512³ | 0.025392 | 0.032899 | 0.772x | pass |
| Direct transpose semantic 384x512x256 | 0.023343 | 0.020402 | 1.144x | pass |
| Wrapper transpose 384x512x256 | 0.023297 | 0.020203 | 1.153x | pass |

Alpha/beta now meets the representative Phase 2 performance target. Transpose
is correctness-clean and no longer pays EDGE overhead, but it remains the active
Phase 2 performance gap.

The current broad/profile evidence is stored under:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_broad_wrapper_mfma_hipsparselt/rocm_vendor_benchmark.json \
  --require-full \
  --require-hipsparselt
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_broad_wrapper_mfma_hipsparselt/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_broad_wrapper_mfma_hipsparselt.md
```

Key rows from that run:

| Row | Generated ms | Vendor ms | Ratio | Correctness |
| --- | ---: | ---: | ---: | --- |
| CTA64 shared 512³ | 0.020573 | 0.025700 | 0.800x | pass |
| CTA128 1024³ | 0.053449 | 0.042599 | 1.255x | pass |
| Public wrapper MFMA 1024³ | 0.057410 | 0.046028 | 1.247x | pass |
| CSR SpMV rowNnz=27 | 0.043661 | 0.019397 | 2.251x | pass |
| hipSPARSELt structured GEMM 512³ | 0.011817 | 0.019310 | 0.612x | pass |

The companion profile directory contains `mfma_resource_metadata.csv`; it
confirms CTA128 is scratch-free (`Scratch_Size=0`, `VGPR_Count=84`) while
CTA128_EDGE is also scratch-free but materially slower (`VGPR_Count=112` and
about `4.005x` versus rocBLAS at 1024³ in the profile matrix). EDGE/tail
performance remains the main MFMA production-hardening gap.

The `20260624_mi300x_edge_tails_profile` fixture records true non-divisible
tail shapes and a matching rocprof resource pass:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_edge_tails_profile/rocm_vendor_benchmark.json \
  --require-full
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_edge_tails_profile/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_edge_tails.md
```

Key rows from that run:

| Row | Generated ms | Vendor ms | Ratio | Correctness |
| --- | ---: | ---: | ---: | --- |
| CTA64_EDGE 1023x1024x2003 | 0.187162 | 0.092749 | 2.018x | pass |
| CTA128_EDGE 1023x1024x2003 | 0.354455 | 0.092947 | 3.814x | pass |
| Wrapper auto 1023x1024x2003 | 0.093014 | 0.093081 | 0.999x | pass |
| CTA64_EDGE 1024x1023x2047 | 0.195389 | 0.108331 | 1.804x | pass |
| CTA128_EDGE 1024x1023x2047 | 0.378784 | 0.108510 | 3.491x | pass |
| Wrapper auto 1024x1023x2047 | 0.108547 | 0.108140 | 1.004x | pass |
| CTA64_EDGE 1025x513x1024 | 0.071904 | 0.060427 | 1.190x | pass |
| CTA128_EDGE 1025x513x1024 | 0.128257 | 0.060246 | 2.129x | pass |
| Wrapper auto 1025x513x1024 | 0.059872 | 0.059853 | 1.000x | pass |

The profile resource summary confirms both EDGE kernels remain scratch-free on
true-tail shapes. CTA64_EDGE reports `LDS_Block_Size=2048`,
`VGPR_Count=32`, and `SGPR_Count=48`; CTA128_EDGE reports
`LDS_Block_Size=16384`, `VGPR_Count=112`, and `SGPR_Count=112`. Public auto
dispatch therefore keeps using rocBLAS for large true-tail mixed-precision
GEMM when the vendor path is available; generated EDGE kernels remain
correctness-validated fallback/diagnostic paths until their edge-path
instruction overhead is reduced.

The follow-up
`20260624_mi300x_edge_tails_store_fastpath_rocm724` fixture validates the
alpha/beta store fast path on ROCm 7.2.4. Correctness and the scratch-free
resource envelope are preserved. Direct EDGE row timings improved only at
noise-level scale versus `20260624_mi300x_edge_tails_profile`
(`0.2%` to `0.6%` on sampled direct EDGE rows), so alpha/beta epilogue
branching is not the dominant EDGE bottleneck. The remaining production work
is still predicate/address/control overhead reduction in the generated EDGE
kernel body, while wrapper auto remains vendor-routed for large true-tail
HIP/gfx942 shapes.

The `20260625_mi300x_edge_tails_predicate_hoist_rocm724` fixture validates the
first CTA64_EDGE predicate-hoist cleanup. The generated source now hoists the
M/N-side row and column predicates out of the K-loop body and emits two explicit
cooperative-load steps instead of the previous `loadIter` loop. Correctness is
unchanged and the profile remains scratch-free (`Scratch_Size=0`,
`VGPR_Count=40`, `SGPR_Count=48` for CTA64_EDGE), but wall-time is effectively
unchanged: direct CTA64_EDGE rows moved by about `+1%` on the sampled true-tail
shapes versus the store-fastpath fixture. This makes predicate hoisting a
source-shape cleanup rather than a throughput fix. The next meaningful EDGE
performance step is structural: split-K/interior-border decomposition or a
different EDGE kernel family, while public auto dispatch remains vendor-routed
for the sampled large true-tail HIP/gfx942 cases.

The `20260625_mi300x_edge_split_rocm724` fixture validates the first opt-in
split EDGE wrapper path:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260625_mi300x_edge_split_rocm724/rocm_vendor_benchmark.json \
  --require-full
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260625_mi300x_edge_split_rocm724/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_edge_split.md
```

The split path is correctness-clean, but slower than the existing single-pass
EDGE path on all sampled true-tail shapes. Forced wrapper MFMA split measured
`3.459x`, `2.936x`, and `2.834x` versus rocBLAS for
`1023x1024x2003`, `1024x1023x2047`, and `1025x513x1024`, respectively. The
corresponding forced wrapper single-pass MFMA rows were `2.129x`, `1.789x`,
and `1.198x`; public `AUTO` remained vendor-routed at `1.001x` to `1.003x`.
The wrapper timing diagnostics show host wall time tracks stream-event time, so
the split regression is device-path/multi-launch work rather than public ABI
submission overhead. Treat split EDGE as a diagnostic path only unless a future
fused true-tail kernel family changes this result.

The `20260625_mi300x_edge_nn_rocm724` fixture validates the fused CTA64 NN-only
fast-tail kernel:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260625_mi300x_edge_nn_rocm724/rocm_vendor_benchmark.json \
  --require-full
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260625_mi300x_edge_nn_rocm724/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_edge_nn.md
```

The row is applicable only to `NN`, `alpha = 1`, `beta = 0`, single-batch
true-tail GEMM calls. It is correctness-clean and scratch-free. The profile
captures a lower-pressure code object (`VGPR_Count=24`, `SGPR_Count=32`,
`Scratch_Size=0`) than generic CTA64_EDGE (`VGPR_Count=40`, `SGPR_Count=48`,
`Scratch_Size=0`), but measured timing only improves generic EDGE by about
`2%` to `4%` on the sampled direct rows:

| Shape | CTA64 EDGE ms | CTA64 EDGE NN ms | rocBLAS ms | AUTO ms |
| --- | ---: | ---: | ---: | ---: |
| 1023x1024x2003 | 0.183973 | 0.178986 | 0.093447 | 0.092904 |
| 1024x1023x2047 | 0.196521 | 0.192181 | 0.108273 | 0.108736 |
| 1025x513x1024 | 0.072629 | 0.069763 | 0.059975 | 0.059781 |

Forced wrapper MFMA now routes these simple true-tail rows through the NN
fast-tail candidate and is correctness-clean, but still trails rocBLAS.
Public AUTO remains vendor-routed for these true-tail shapes.

The `20260624_mi300x_vendor_wrapper_cache` fixture records the first
Float32 vendor-wrapper cleanup check after rocBLAS pointer-mode and stream
state were cached in the HIP `LibraryOps` adapter:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_vendor_wrapper_cache/rocm_vendor_benchmark.json \
  --require-full
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_vendor_wrapper_cache/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_vendor_wrapper_cache.md
```

Key rows from that run:

| Row | Generated ms | Vendor ms | Ratio | Correctness |
| --- | ---: | ---: | ---: | --- |
| F32 vendor wrapper 512³ | 0.008297 | 0.008317 | 0.998x | pass |
| F32 vendor wrapper 1024³ | 0.030552 | 0.025679 | 1.190x | pass |
| F16/F32 wrapper MFMA 512³ | 0.021085 | 0.026753 | 0.788x | pass |
| F16/F32 wrapper MFMA 1024³ | 0.057701 | 0.047498 | 1.215x | pass |

This fixture shows that rocBLAS handle-state caching is correctness-safe but
does not remove the large-shape Float32 vendor-wrapper overhead. Further work
should instrument the public ABI path instead of repeatedly tuning rocBLAS
handle setup.

The `20260624_mi300x_cta128_evidence_after_cache` fixture reruns the focused
CTA64/CTA128/direct/wrapper matrix after the same cleanup:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_cta128_evidence_after_cache/rocm_vendor_benchmark.json \
  --require-full
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_cta128_evidence_after_cache/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_cta128_after_cache.md
```

Key rows from that run:

| Row | Generated ms | Vendor ms | Ratio | Correctness |
| --- | ---: | ---: | ---: | --- |
| CTA64 shared 512³ | 0.020528 | 0.025649 | 0.800x | pass |
| CTA128 1024³ | 0.053284 | 0.042698 | 1.248x | pass |
| Public wrapper MFMA 1024³ | 0.053464 | 0.042794 | 1.249x | pass |
| CTA64 shared 512x1024x1024 | 0.039473 | 0.042325 | 0.933x | pass |
| Public wrapper MFMA 512x1024x1024 | 0.039701 | 0.042443 | 0.935x | pass |

The wrapper/direct relationship is again within measurement noise in the
focused fixture. The larger wrapper row seen in the broad matrix should be read
as broad-fixture variance or queue-state sensitivity, not as a confirmed
fast-path regression.

The 2026-06-23 fixture adds the HIP/gfx942 MFMA K-loop GEMM row to the broad
matrix and records the same machine class with ROCm 7.2.4:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/rocm_vendor_benchmark_mi300x_20260623_rocm724_mfma.json \
  --require-full \
  --require-hipsparselt
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/rocm_vendor_benchmark_mi300x_20260623_rocm724_mfma.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_20260623_rocm724_mfma.md
```

This fixture is the first durable correctness record for
`GEMM_F16_MFMA` at 128³, 512³, and 1024³. It should be read as an
implementation-validation and tuning artifact for the experimental MFMA path;
the row is still narrower than a production arbitrary-shape GEMM replacement.
It predates any portable F16 denominator row, so it does not by itself evaluate
MFMA speedup against a portable F16-input GEMM kernel.

The follow-up 2026-06-23 fixture added historical portable F16-output rows to
the same broad matrix:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/rocm_vendor_benchmark_mi300x_20260623_rocm724_mfma_portable_f16.json \
  --require-full \
  --require-hipsparselt
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/rocm_vendor_benchmark_mi300x_20260623_rocm724_mfma_portable_f16.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_20260623_rocm724_mfma_portable_f16.md
```

That report is retained as historical evidence, but it is not the final R5
denominator because it compares F16-output portable GEMM against F32-output
MFMA. Its measured legacy comparison was:

| Shape | Portable F16 mean ms | MFMA mean ms | MFMA speedup over portable F16 | R3 gate |
| --- | ---: | ---: | ---: | --- |
| 128³ | 0.006002 | 0.006032 | 0.995x | informational |
| 512³ | 0.027301 | 0.019391 | 1.408x | below 5x |
| 1024³ | 0.177962 | 0.107498 | 1.655x | below 10x |

The result confirms MFMA correctness and a measurable speedup over the legacy
portable F16-output kernel at larger sizes, but it does not satisfy the R5
apples-to-apples denominator requirement. Production hardening remains required
before any MFMA speedup claim should be made.

The CTA direct follow-up fixture records the first apples-to-apples
`GEMM_F16_PORTABLE_F32OUT` denominator together with the historical K-loop,
CTA64 direct-load, and CTA128 shared-memory MFMA rows:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260623_mi300x_cta64_direct/rocm_vendor_benchmark.json \
  --require-full \
  --require-hipsparselt
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260623_mi300x_cta64_direct/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_cta64_direct.md
```

Key rows from that run on MI300X VF / ROCm 7.2.4:

| Shape | Portable F16/F32-out ms | K-loop MFMA ms | CTA64 direct ms | CTA128 shared ms |
| --- | ---: | ---: | ---: | ---: |
| 128³ | 0.006199 | 0.006431 | 0.012316 | 0.014937 |
| 512³ | 0.027261 | 0.019733 | 0.044474 | 0.051593 |
| 1024³ | 0.181003 | 0.107697 | 0.108873 | 0.147030 |

The paired rocprof trace in the same hardware-run directory confirms the
resource-pressure diagnosis: K-loop and CTA64 direct both compile with
`LDS_Block_Size=0`, `Scratch_Size=0`, and `VGPR_Count=40`; CTA128 compiles with
`LDS_Block_Size=16384`, `Scratch_Size=12`, and `VGPR_Count=128`. CTA64 direct
therefore fixes the CTA128 scratch/register-pressure issue, but further
throughput work is still required before it can become the production GEMM row.

The CTA64 shared-memory follow-up fixture keeps the same low-pressure CTA64
register geometry but stages a 64x8 A panel and an 8x64 B panel into LDS for
each K step:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260623_mi300x_cta64_shared/broad/rocm_vendor_benchmark.json \
  --require-full \
  --require-hipsparselt
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260623_mi300x_cta64_shared/broad/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_cta64_shared.md
```

Key rows from that run on MI300X VF / ROCm 7.2.4:

| Shape | Portable F16/F32-out ms | K-loop MFMA ms | CTA64 direct ms | CTA64 shared ms | CTA128 shared ms | rocBLAS F16/F32 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128³ | 0.006034 | 0.006118 | 0.012114 | 0.006281 | 0.014600 | 0.023803 |
| 512³ | 0.027395 | 0.019503 | 0.044245 | 0.020595 | 0.051218 | 0.025759 |
| 1024³ | 0.179902 | 0.107676 | 0.108166 | 0.059890 | 0.146315 | 0.042812 |

The paired rocprof trace records the resource signature:

| Kernel | LDS bytes | Scratch bytes | VGPR | Workgroup |
| --- | ---: | ---: | ---: | --- |
| `navatala_transformer_tiled_gemm_f16_mfma_k_loop` | 0 | 0 | 40 | 64x1x1 |
| `navatala_transformer_tiled_gemm_f16_mfma_cta64_direct` | 0 | 0 | 40 | 256x1x1 |
| `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared` | 2048 | 0 | 40 | 256x1x1 |
| `navatala_transformer_tiled_gemm_f16_mfma_cta128` | 16384 | 12 | 128 | 256x1x1 |

CTA64 shared therefore closes the CTA64 direct memory-coalescing gap without
reintroducing scratch. It is still not a production arbitrary-shape GEMM
replacement: at 1024³ it is about `1.399x` slower than rocBLAS F16/F32, and it
still supports only tile-divisible NN, alpha=1, beta=0 shapes.

The CTA64 early-barrier / padded-LDS follow-up fixture records two additional
low-pressure tuning diagnostics:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260623_mi300x_cta64_early_barrier/broad/rocm_vendor_benchmark.json \
  --require-full \
  --require-hipsparselt
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260623_mi300x_cta64_early_barrier/broad/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_cta64_early_barrier.md
```

Key rows from that run on MI300X VF / ROCm 7.2.4:

| Shape | CTA64 shared ms | CTA64 early-barrier ms | CTA64 padded ms | rocBLAS F16/F32 ms |
| --- | ---: | ---: | ---: | ---: |
| 128³ | 0.006338 | 0.006396 | 0.008066 | 0.023767 |
| 512³ | 0.020577 | 0.020679 | 0.027433 | 0.025706 |
| 1024³ | 0.060042 | 0.060080 | 0.073456 | 0.043034 |

The paired rocprof trace records:

| Kernel | LDS bytes | Scratch bytes | VGPR | Accum VGPR | Workgroup |
| --- | ---: | ---: | ---: | ---: | --- |
| `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared` | 2048 | 0 | 40 | 0 | 256x1x1 |
| `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier` | 2048 | 0 | 40 | 0 | 256x1x1 |
| `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded` | 2560 | 0 | 44 | 4 | 256x1x1 |

The early-barrier variant preserves the clean CTA64 resource envelope but is
performance-neutral. The padded-LDS variant is correct and spill-free but slower
because it increases LDS/register pressure. These rows are retained as tuning
evidence; the best measured generated MFMA prototype remains
`GEMM_F16_MFMA_CTA64_SHARED`.

The R6 CTA64 pipelined follow-up fixture records the two-slot staged-panel
contract after the typed shared-panel/vector-copy IR landed:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260623_mi300x_cta64_pipelined/broad/rocm_vendor_benchmark.json \
  --require-full \
  --require-hipsparselt
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260623_mi300x_cta64_pipelined/broad/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_cta64_pipelined.md
```

Key rows from that run on MI300X VF / ROCm 7.2.4:

| Shape | CTA64 shared ms | CTA64 early-barrier ms | CTA64 padded ms | CTA64 pipelined ms | rocBLAS F16/F32 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128³ | 0.006306 | 0.006314 | 0.008092 | 0.009071 | 0.023759 |
| 512³ | 0.020569 | 0.020548 | 0.027301 | 0.031763 | 0.025713 |
| 1024³ | 0.059997 | 0.059885 | 0.073313 | 0.084913 | 0.042828 |

The paired resource metadata records:

| Kernel | LDS bytes | Scratch bytes | SGPR | VGPR | VGPR spill | Workgroup |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared` | 2048 | 0 | 23 | 40 | 0 | 256x1x1 |
| `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier` | 2048 | 0 | 23 | 40 | 0 | 256x1x1 |
| `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded` | 2192 | 0 | 23 | 42 | 0 | 256x1x1 |
| `navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined` | 4096 | 0 | 70 | 51 | 0 | 256x1x1 |

The pipelined row is therefore validated for correctness and no-scratch
compilation, but it is not yet a performance win. That first fixture used the
R6 copy-group contract with synchronous HIP lowering, so the two-slot structure
added register and LDS pressure without hiding global-memory latency.

The direct global-to-LDS follow-up fixture records the same R6 contract after
HIP lowering was changed to `buffer_load_dword ... lds` for
`CopyGlobalToShared` plus `s_waitcnt vmcnt(0)`/barrier wait semantics for
`AwaitCopyGroup`:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260623_mi300x_direct_lds/broad/rocm_vendor_benchmark.json \
  --require-full \
  --require-hipsparselt
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260623_mi300x_direct_lds/broad/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_direct_lds.md
```

Key rows from that run on MI300X VF / ROCm 7.2.4:

| Shape | CTA64 shared ms | CTA64 pipelined direct-LDS ms | rocBLAS F16/F32 ms |
| --- | ---: | ---: | ---: |
| 128³ | 0.006269 | 0.008582 | 0.024025 |
| 512³ | 0.020553 | 0.027703 | 0.026632 |
| 1024³ | 0.059940 | 0.060183 | 0.043066 |

The paired resource metadata records:

| Kernel | LDS bytes | Scratch bytes | SGPR | VGPR | VGPR spill | Workgroup |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared` | 2048 | 0 | 23 | 40 | 0 | 256x1x1 |
| `navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined` | 4096 | 0 | 94 | 73 | 0 | 256x1x1 |

The direct-LDS lowerer is therefore functionally validated and remains
scratch-free, but it is not the production GEMM row yet. It improves the old
synchronous pipelined 1024³ timing from `0.084913 ms` to `0.060183 ms`, roughly
matching CTA64 shared at that size, but it is slower than CTA64 shared at
512³ and carries substantially higher SGPR/VGPR pressure. The next GEMM tuning
step should reduce direct-LDS register pressure and scheduling overhead, not
claim that the pipelined path has displaced the CTA64 shared prototype.

The CTA128 evidence fixture records the current large-shape candidate after the
helper-signature and launch-bounds cleanup:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260623_mi300x_cta128_evidence/rocm_vendor_benchmark.json \
  --require-full
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260623_mi300x_cta128_evidence/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_cta128_evidence.md
```

Key rows from that run on MI300X VF / ROCm 7.2.4:

| Shape | Portable F16/F32-out ms | CTA64 shared ms | CTA64 pipelined ms | CTA128 ms | rocBLAS F16/F32 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128x128x128 | 0.006003 | 0.006310 | 0.007229 | 0.008454 | 0.023802 |
| 512x512x512 | 0.027258 | 0.020572 | 0.022739 | 0.026223 | 0.025666 |
| 1024x1024x1024 | 0.181677 | 0.060056 | 0.061617 | 0.053518 | 0.042751 |
| 1024x512x1024 | 0.096475 | 0.053069 | 0.055340 | 0.049700 | 0.042576 |
| 512x1024x1024 | 0.094872 | 0.039428 | 0.043786 | 0.049743 | 0.042378 |
| 2048x512x1024 | 0.183262 | 0.058979 | 0.059979 | 0.053395 | 0.042729 |

The paired profile summary records the current CTA128 resource signature:

| Kernel | LDS bytes | Scratch bytes | VGPR | Accum VGPR | SGPR | Workgroup |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `navatala_transformer_tiled_gemm_f16_mfma_cta128` | 16384 | 0 | 84 | 68 | 112 | 256x1x1 |

This supersedes the older CTA128 `Scratch_Size=12` fixtures for current-code
resource diagnosis. The performance result is shape-dependent: CTA64 shared
remains the medium-shape candidate, while CTA128 is the best generated row for
1024³ and most tall/large rectangular cases. The public runtime exposes
`navatala_gpu_gemm_f16_f32` as the shape-aware MFMA dispatch entrypoint for
HIP/gfx942 F16-input/F32-output GEMM. The benchmark rows in this fixture still
measure the tile-divisible Phase-0 prototype kernels; wrapper-level correctness
for edge sizes, alpha/beta, transpose, and batched calls is covered by the FFI
test fixture.

The 2026-06-24 CTA128 edge-wrapper fixture reruns the focused CTA128 evidence
matrix after the public FFI wrapper was extended to dispatch the edge-capable
CTA64/CTA128 kernels:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_cta128_edge_wrapper/rocm_vendor_benchmark.json \
  --require-full
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_cta128_edge_wrapper/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_cta128_edge_wrapper.md
```

The same run was paired with a full HIP CTest pass on MI300X VF / ROCm 7.2.4.
The FFI test explicitly executed:

```text
running F16/F32 GEMM FFI suite: default backend=hip
running F16/F32 GEMM FFI suite: forced-mfma-cta64 backend=hip
running F16/F32 GEMM FFI suite: forced-mfma-cta128 backend=hip
```

That test covers small edge-sized GEMMs, alpha/beta accumulation, NT/TN/TT
transpose modes, strided batching, broadcast input batching, and rejection of
overlapping output batches. Treat this as wrapper-correctness evidence; the
performance report remains a Phase-0 timing fixture for the tile-divisible MFMA
prototype rows.

The paired profile summary confirms CTA128 is still scratch-free after the
wrapper changes:

| Kernel | LDS bytes | Scratch bytes | VGPR | Accum VGPR | SGPR | Workgroup |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `navatala_transformer_tiled_gemm_f16_mfma_cta128` | 16384 | 0 | 84 | 68 | 112 | 256x1x1 |

The 2026-06-24 packed-parameter wrapper fixture records the first direct timing
evidence for the public F16-input/F32-output MFMA wrapper row after the runtime
was changed to pack scalar launch parameters into a single device buffer:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_mfma_packed_params_standard/rocm_vendor_benchmark.json \
  --require-full
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_mfma_packed_params_standard/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_wrapper_mfma_packed_params_standard.md
```

Key wrapper rows from that run:

| Row | Shape | Generated mean ms | Vendor mean ms | Ratio | Max abs error |
| --- | --- | ---: | ---: | ---: | ---: |
| `GEMM_F16_F32_WRAPPER_MFMA` | 512³ | 0.051632 | 0.026255 | 1.967x | 1.013279e-06 |
| `GEMM_F16_F32_WRAPPER_MFMA` | 513x511x257 | 0.033700 | 0.039212 | 0.859x | 7.152557e-07 |

This is wrapper-dispatch evidence, not a raw microkernel row. Timing includes
the public C ABI dispatch and the final synchronization needed to keep the
temporary packed parameter block alive. The packed-parameter runtime path
replaces the earlier per-scalar temporary-buffer implementation, which measured
about `0.241 ms` for the same 512³ wrapper row in the pre-fix standard run.

The focused CTA128 evidence matrix was rerun with the same public wrapper row
enabled, producing:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_mfma_cta128_evidence/rocm_vendor_benchmark.json \
  --require-full
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_wrapper_mfma_cta128_evidence/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_wrapper_mfma_cta128_evidence.md
```

Key public-wrapper rows from that focused run:

| Row | Shape | Generated mean ms | Vendor mean ms | Ratio | Max abs error |
| --- | --- | ---: | ---: | ---: | ---: |
| `GEMM_F16_F32_WRAPPER_MFMA` | 128³ | 0.027570 | 0.024648 | 1.119x | 1.19209e-07 |
| `GEMM_F16_F32_WRAPPER_MFMA` | 512³ | 0.048698 | 0.026163 | 1.861x | 1.01328e-06 |
| `GEMM_F16_F32_WRAPPER_MFMA` | 1024³ | 0.190138 | 0.043639 | 4.357x | 2.14577e-06 |
| `GEMM_F16_F32_WRAPPER_MFMA` | 1024x512x1024 | 0.141080 | 0.043086 | 3.274x | 1.3113e-06 |
| `GEMM_F16_F32_WRAPPER_MFMA` | 512x1024x1024 | 0.079857 | 0.042973 | 1.858x | 2.14577e-06 |
| `GEMM_F16_F32_WRAPPER_MFMA` | 2048x512x1024 | 0.195592 | 0.043495 | 4.497x | 1.3113e-06 |

All focused wrapper rows passed correctness. The larger tile-divisible shapes
also show that the current public wrapper still carries more overhead than the
raw MFMA micro-benchmark rows. Treat this fixture as release evidence for
wrapper correctness and as the baseline for the next runtime optimization pass:
cached launch/program state, a by-value scalar launch ABI, or an async lifetime
model that removes the final synchronization around the packed parameter block.

The 2026-06-24 broad MFMA + hipSPARSELt fixture keeps the wider AMD-facing
comparison matrix in one durable artifact:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_broad_mfma_hipsparselt/rocm_vendor_benchmark.json \
  --require-full \
  --require-hipsparselt
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_broad_mfma_hipsparselt/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_broad_mfma_hipsparselt.md
```

Key rows from that run:

| Row | Shape | Generated mean ms | Vendor mean ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| `GEMM_F16_MFMA_CTA64_SHARED` | 512³ | 0.020594 | 0.025680 | 0.802x |
| `GEMM_F16_MFMA_CTA64_SHARED` | 1024³ | 0.059771 | 0.042855 | 1.395x |
| `GEMM_F16_MFMA_CTA128` | 512³ | 0.026327 | 0.026231 | 1.004x |
| `GEMM_F16_MFMA_CTA128` | 1024³ | 0.053358 | 0.042992 | 1.241x |
| `CSR_SPMV_F32` | rows=262144,rowNnz=27 | 0.043721 | 0.020461 | 2.137x |
| `HIPSPARSELT_STRUCTURED_GEMM_F16` | 512³, 50% sparsity | 0.012567 | 0.022986 | 0.547x |

This fixture demonstrates that the public benchmark harness now exercises
rocBLAS, rocSPARSE, generated MFMA rows, and hipSPARSELt structured sparse GEMM
on the same ROCm installation. The hipSPARSELt row computes a structured-sparse
operation and must not be compared directly with dense MFMA/rocBLAS GEMM rows.

The follow-up 2026-06-24 v2-SpMV fixture reruns the same broad matrix after the
benchmark harness was migrated from deprecated `rocsparse_spmv` to
`rocsparse_v2_spmv`:

```bash
python3 scripts/validate_rocm_benchmark_json.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_broad_v2spmv_mfma_hipsparselt/rocm_vendor_benchmark.json \
  --require-full \
  --require-hipsparselt
python3 scripts/render_rocm_benchmark_report.py \
  benchmarks/fixtures/hardware_runs/20260624_mi300x_broad_v2spmv_mfma_hipsparselt/rocm_vendor_benchmark.json \
  --output /tmp/rocm_vendor_benchmark_mi300x_broad_v2spmv_mfma_hipsparselt.md
```

This is the preferred current broad fixture for SpMV vendor comparisons because
it uses the non-deprecated rocSPARSE v2 API. Key rows:

| Row | Shape | Generated mean ms | Vendor mean ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| `GEMM_F16_MFMA_CTA64_SHARED` | 512³ | 0.020580 | 0.025724 | 0.800x |
| `GEMM_F16_MFMA_CTA128` | 1024³ | 0.053409 | 0.042726 | 1.250x |
| `CSR_SPMV_F32` | rows=262144,rowNnz=15 | 0.024228 | 0.010192 | 2.377x |
| `CSR_SPMV_F32` | rows=262144,rowNnz=27 | 0.044558 | 0.019642 | 2.269x |
| `HIPSPARSELT_STRUCTURED_GEMM_F16` | 512³, 50% sparsity | 0.012455 | 0.019395 | 0.642x |

The rowNnz=27 result remains within the SpMV tuning gate (`<= 3x`) with the v2
rocSPARSE baseline. Low-rowNnz large-row-count SpMV is still slower than
rocSPARSE (`rows=1048576,rowNnz=7` is 2.425x), so short-row large-matrix tuning
is a separate follow-up rather than a reason to reopen the high-rowNnz fix.

The older `rocm_vendor_benchmark_mi300x_20260619.json` fixture is retained as a
historical pre-provenance baseline. Validator warnings about missing `rocminfo`
and `kernelClass` on that older fixture are expected because those schema fields
were added after the run was captured.

Before running the vendor benchmark on CI, the ROCm workflow also runs:

```bash
./scripts/run_rocm_runtime_smoke.sh
```

That script builds the public runtime with HIP enabled and runs the core CTest
smoke set with `GPU_RUNTIME_BACKEND=hip`. The default test regex is:

```text
test_device_selection|test_events|test_graphs|test_kernel_execution|test_program_cache|manifest_conformance|rocm_benchmark_json_fixture_conformance|rocm_benchmark_report_fixture_render
```

Override it with `NAVATALA_GPU_ROCM_SMOKE_REGEX` when a runner needs a narrower
or broader gate.

Use `NAVATALA_GPU_ROCM_BENCH_MATRIX=broad` with the wrapper, or pass
`--matrix broad` directly, to run the multi-size matrix. The broad matrix keeps
the same operation set but emits several shapes per operation so scale-sensitive
issues are visible in one JSON report. Use
`NAVATALA_GPU_ROCM_BENCH_MATRIX=cta128-evidence` or
`--matrix cta128-evidence` for the CTA128 hardening matrix: it records the
portable F16/F32-output denominator, CTA64 shared, CTA64 pipelined, and CTA128
rows for square and representative rectangular tile-divisible shapes.
`NAVATALA_GPU_ROCM_BENCH_QUICK=1` overrides the matrix selection and emits the
smoke-only `quick` matrix.
Set `NAVATALA_GPU_ROCM_BENCH_INCLUDE_MFMA=1` or pass
`--include-mfma-benchmark` to add the experimental HIP/gfx942 MFMA F16/F32
GEMM rows. The wrapper builds those rows only when requested because the
generated kernels use ROCm 6.2+ matrix intrinsics and are intended for
MI300/gfx942-class hardware.

The opt-in MFMA rows currently distinguish two implementation stages:

| Operation | Kernel | Scope |
| --- | --- | --- |
| `GEMM_F16_MFMA` | `navatala_transformer_tiled_gemm_f16_mfma_k_loop` | One 32x32 output tile per wave, arbitrary K multiple of 8, no edge/alpha/beta/transpose support. Historical prototype row. |
| `GEMM_F16_MFMA_CTA64_DIRECT` | `navatala_transformer_tiled_gemm_f16_mfma_cta64_direct` | Low-pressure 64x64 CTA diagnostic prototype: four wave64s per workgroup, one accumulator per wave, no shared-memory staging, tile-divisible NN only, alpha=1, beta=0. This row is intended to verify whether reducing register pressure/scratch improves MI300/gfx942 performance. |
| `GEMM_F16_MFMA_CTA64_SHARED` | `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared` | Low-pressure 64x64 CTA plus 2 KB LDS staging: four wave64s per workgroup, one accumulator per wave, shared panel staging, tile-divisible NN only, alpha=1, beta=0. This row is the current best measured tuned MFMA prototype. |
| `GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER` | `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier` | Diagnostic CTA64 shared variant with the LDS overwrite-safety barrier moved immediately after fragment loads. Correct and zero-scratch, but neutral versus CTA64 shared on MI300X. |
| `GEMM_F16_MFMA_CTA64_SHARED_PADDED` | `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded` | Diagnostic CTA64 shared variant with padded LDS strides (`64x9` A, `8x65` B). Correct and zero-scratch, but slower than CTA64 shared on MI300X. |
| `GEMM_F16_MFMA_CTA64_PIPELINED` | `navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined` | R6 two-slot staged-panel row: four wave64s per workgroup, one accumulator per wave, typed shared panels, `b16` copy requests, exact dynamic copy-group matching, tile-divisible NN only, alpha=1, beta=0. HIP/gfx942 now lowers aligned panel copies through direct global-to-LDS dword loads and waits with `s_waitcnt vmcnt(0)`, but this row is still a tuning/resource-validation path rather than the production GEMM dispatch. |
| `GEMM_F16_MFMA_CTA128` | `navatala_transformer_tiled_gemm_f16_mfma_cta128` | Phase-0 128x128x32 CTA prototype with shared-memory staging, four wave64s per workgroup, tile-divisible NN only, alpha=1, beta=0. This row measures the shared-memory production-hardening path on MI300/gfx942 hardware. |

The public FFI wrapper dispatches different edge-capable kernels for production
calls: `navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge` and
`navatala_transformer_tiled_gemm_f16_mfma_cta128_edge`. Those kernels cover
tail tiles, alpha/beta, transpose flags, and batching. They are validated by
the runtime FFI test. The opt-in `GEMM_F16_F32_WRAPPER_MFMA` row additionally
times the public `navatala_gpu_gemm_f16_f32` C ABI with
`NAVATALA_GPU_GEMM_IMPL=mfma`; it is wrapper evidence and includes runtime
scalar-parameter setup/synchronization overhead, so compare it separately from
the raw tile-divisible micro-benchmark rows above.

The broad matrix emits `GEMM_F16_PORTABLE_F32OUT` at 128³, 512³, and 1024³.
Those rows run the generated `navatala_transformer_tiled_gemm_f16_f32_out`
kernel and provide the apples-to-apples denominator for MFMA speedup tracking.
For a matching shape, compute:

```text
MFMA speedup over portable F16/F32-output =
  GEMM_F16_PORTABLE_F32OUT.generatedMeanMs / GEMM_F16_MFMA_CTA64_DIRECT.generatedMeanMs
  GEMM_F16_PORTABLE_F32OUT.generatedMeanMs / GEMM_F16_MFMA_CTA64_SHARED.generatedMeanMs
  GEMM_F16_PORTABLE_F32OUT.generatedMeanMs / GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER.generatedMeanMs
  GEMM_F16_PORTABLE_F32OUT.generatedMeanMs / GEMM_F16_MFMA_CTA64_SHARED_PADDED.generatedMeanMs
  GEMM_F16_PORTABLE_F32OUT.generatedMeanMs / GEMM_F16_MFMA_CTA64_PIPELINED.generatedMeanMs
  GEMM_F16_PORTABLE_F32OUT.generatedMeanMs / GEMM_F16_MFMA_CTA128.generatedMeanMs
```

The gate is not directly evaluable unless both rows are present for the same
shape in the same-host report.

The intended MFMA runtime policy is shape-aware: CTA64 shared remains the
medium-shape candidate, while CTA128 is the large-shape candidate. The public
`navatala_gpu_gemm_f16_f32` wrapper follows that policy for HIP/gfx942 when
`NAVATALA_GPU_GEMM_IMPL` is `auto` or `mfma`, using the edge-capable
CTA64/CTA128 kernels for tail tiles, alpha/beta, transpose, and strided-batch
calls. The benchmark rows above intentionally continue to measure the
tile-divisible Phase-0 kernels for clean historical comparisons.
When publishing a CTA128 advancement, include both the timing JSON/report and
the `profile/mfma_resource_metadata.csv` emitted by
`./scripts/run_rocm_vendor_benchmarks.sh --matrix cta128-evidence
--include-mfma-benchmark --profile-cta128`; CTA128 must remain
`Scratch_Size=0`.

Near-term tuning work must keep the broad matrix explicit enough to diagnose
dispatch thresholds. For CSR SpMV this means reporting at least:

| rows | rowNnz | Purpose |
| ---: | ---: | --- |
| 16384 | 7 | small short-row branch |
| 262144 | 7 | large short-row branch |
| 1048576 | 7 | memory-bandwidth short-row branch |
| 262144 | 15 | threshold transition branch |
| 262144 | 27 | CFD-like medium/high-row branch |

The 2026-06-22 MI300X reference fixture includes both rowNnz=15 and rowNnz=27.
The rowNnz=27 row exercises the subgroup-per-row path and is the primary
acceptance row for the adaptive-SpMV tuning ticket.

## Timing Semantics

The benchmark reports back-to-back throughput mean per launch, not isolated
single-launch latency. Each result is measured with HIP events around a warm
sequence of repeated launches and divided by `iterations`. The JSON records
this as:

```json
"timingMode": "back_to_back_throughput_mean_per_launch"
```

Quick mode is a smoke check only. In particular, the quick GEMM shape is small
enough that rocBLAS launch/setup overhead can dominate, so quick-mode numbers
should not be used as performance claims. Use non-quick runs, full logs, and the
hardware/runtime metadata when publishing benchmark comparisons.

The JSON also records the matrix set as `matrix: "quick" | "standard" |
"broad" | "cta128-evidence" | "cfd-fixture"`. Duplicate operation names are
allowed when their shapes differ.

Benchmark reports also record implementation selection metadata:

```json
{
  "kernelClass": "scalar|mfma_f16|wmma_tf32|vendor_library",
  "implementationKind": "portable_kernel|tuned_kernel|vendor_library",
  "tuningPath": "thread_per_row|subgroup_per_row|portable_f16_tiled|portable_f16_f32out_tiled|adaptive|vendor_dispatch|hip_mfma_gfx942_32x32x8_f16_f32_k_loop|hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct|hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared|hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier|hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded|hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined|hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge|hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_semantic|hip_mfma_gfx942_128x128x32_f16_f32_cta128|hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge|hip_mfma_gfx942_128x128x32_f16_f32_cta128_semantic|hip_mfma_gfx942_wrapper_dispatch",
  "spmvRowNnzThreshold": 16,
  "vendorDispatchSelected": false
}
```

`kernelClass` distinguishes scalar portable kernels from future matrix-intrinsic
rows such as `mfma_f16` and from vendor-library rows. Reports also carry a
`rocminfo` object with the active GCN ISA, HIP runtime version, HIP driver
version, and a compact `rocminfo` summary so dated fixtures preserve the
hardware/runtime provenance needed for same-host comparisons.

Wrapper rows may additionally carry host-side timing diagnostics:

```json
{
  "hostSubmitMeanMs": 0.0,
  "hostWallMeanMs": 0.0,
  "hostSubmitOverGeneratedRatio": 0.0,
  "hostWallOverGeneratedRatio": 0.0
}
```

`hostSubmitMeanMs` measures CPU wall time to invoke the public wrapper repeatedly
and enqueue work, excluding the final event wait. `hostWallMeanMs` includes that
final wait. Compare these with `generatedMeanMs`, which remains the stream-event
mean, to separate wrapper submission overhead from device-side work.

The validator keeps these fields backward-compatible with existing
`navatala_gpu.rocm_vendor_benchmark.v1` fixtures because the 2026-06-19 MI300X
reference run predates the adaptive-SpMV metadata and the later
`rocminfo`/`kernelClass` provenance fields.

## Current Operation Coverage

| Operation | Generated HIP kernel | Vendor baseline | Purpose |
| --- | --- | --- | --- |
| AXPY F32 | `navatala_sparse_axpy_f32` | `rocblas_saxpy` | Solver vector-update sanity and memory-bandwidth baseline. |
| GEMM F32 | `navatala_transformer_tiled_gemm_f32` | `rocblas_sgemm` | Dense compute comparison against a tuned Level-3 BLAS path. |
| GEMM F16 portable F32 output | `navatala_transformer_tiled_gemm_f16_f32_out` | `rocblas_gemm_ex` with F16 input/F32 output/F32 accumulation | Portable denominator for MFMA speedup tracking. |
| GEMM F16 MFMA | `navatala_transformer_tiled_gemm_f16_mfma_k_loop` | `rocblas_gemm_ex` with F16 inputs and F32 accumulation | Opt-in dense matrix-intrinsic row for MI300/gfx942 tuning. |
| CSR SpMV F32 | adaptive `navatala_graph_spmv_weighted_f32` / `navatala_graph_spmv_weighted_subgroup_f32` | `rocsparse_spmv` | Sparse baseline relevant to graph, CFD, AMG, and iterative solvers. |
| Structured sparse GEMM F16 | hipSPARSELt prune/compress/matmul path | `rocblas_hgemm` dense pruned-A reference | Opt-in structured-sparse vendor path for `SparseLt_StructuredMatmul`. |

These operations were chosen because they are simple enough to benchmark
reliably, touch both dense and sparse ROCm libraries, and map to workloads
that users expect to compare against rocBLAS/rocSPARSE.

The generated transformer GEMM kernels are portable fallback kernels. Current
generated sources use shared/threadgroup 16x16 tiling, but the ROCm benchmark
must still treat them separately from vendor dispatch and from future
MFMA/WMMA-style tuned kernels. A fresh post-tuning run is required before making
any performance claim for the portable fallback path.

The opt-in `GEMM_F16_MFMA*` rows exercise Navatala HIP MFMA kernels. They are
intentionally narrower than the public FFI wrapper path: matrix sizes must be
multiples of the timing-row tile constraints, and the timing rows keep
alpha=1/beta=0 NN semantics so same-host comparisons stay stable. Use these
rows as matrix-intrinsic correctness and tuning signals, not as general rocBLAS
replacement claims. The CTA64 direct row is specifically a
register-pressure/scratch-reduction experiment relative to the shared-memory
CTA128 row. The public wrapper dispatch has broader correctness coverage through
the FFI tests, but it should not be marketed as a full rocBLAS replacement until
larger edge/transpose/batched benchmark gates are added.

The `GEMM_F16_PORTABLE_F32OUT` row is not a tuned-kernel claim. It exists so
MFMA reports can compare against a portable F16-input/F32-output tiled kernel
in the same benchmark run. This keeps the denominator aligned with the MFMA row,
which also writes F32 output. Use it for MFMA speedup tracking, not for
vendor-dispatch claims.

## hipSPARSELt Position

hipSPARSELt is detected by CMake and reported in benchmark JSON. By default,
the report remains capability-only:

```json
"hipSPARSELtMode": "capability_reporting_only"
```

Set `NAVATALA_GPU_ROCM_BENCH_INCLUDE_HIPSPARSELT=1` or pass
`--include-hipsparselt-benchmark` to add a real hipSPARSELt row. The JSON then
reports:

```json
"hipSPARSELtMode": "vendor_benchmark"
```

The row uses hipSPARSELt prune/check/compress/matmul for a legal structured
sparse A operand and compares the result against `rocblas_hgemm` on the pruned
dense A matrix. On MI300/gfx942 the supported path is FP16 input/output with
FP32 compute. Float32 structured sparse matmul is not a supported hipSPARSELt
path on this hardware, so the benchmark deliberately does not emit an F32
hipSPARSELt row.

This is still benchmark coverage, not a general runtime-dispatch claim. A
production `SparseLt_StructuredMatmul` runtime path still needs:

- dtype/layout/tile legality checks against the installed hipSPARSELt version;
- conversion or rejection rules for non-conforming dense/pruned/compressed
  sparse operands;
- manifest metadata marking wrapper/orchestrator operation paths as
  `vendor_dispatch` only for supported operation/backend combinations;
- wrapper/runtime dispatch integration that preserves the same constraints.

## GEMM Vendor Dispatch Position

The GEMM rows in this benchmark currently compare the portable generated
`navatala_transformer_tiled_gemm_f32` kernel against rocBLAS. That is useful
for measuring the portable fallback gap, but it is not the intended final
performance path for public CUDA/HIP GEMM wrappers.

Near-term vendor-dispatch work should benchmark both paths separately:

- `portable_kernel`: generated tiled GEMM fallback/reference kernel.
- `vendor_library`: rocBLAS/cuBLAS dispatch through `LibraryOp.BLAS_GEMM` when
  dtype, layout, transpose flags, and shape threshold allow it.

The public C ABI `navatala_gpu_gemm_f32` now has a real-runtime CUDA/HIP
vendor-dispatch path controlled by `NAVATALA_GPU_GEMM_VENDOR_MODE` and
`NAVATALA_GPU_GEMM_VENDOR_THRESHOLD_FLOPS`. Existing `GEMM_F32` rows continue
to measure the portable generated tiled kernel against rocBLAS.

`NAVATALA_GPU_GEMM_VENDOR_MODE` accepts:

- `auto` / `1` / `true`: use vendor BLAS when support and threshold checks pass,
  otherwise fall back.
- `vendor` / `always` / `force`: require vendor BLAS and fail loudly if it is
  unavailable.
- `portable` / `reference` / `fallback` / `0` / `false`: disable vendor BLAS and
  use the available fallback/reference path.

`NAVATALA_GPU_GEMM_IMPL=auto|vendor|mfma|portable` is a higher-precedence
wrapper implementation selector for validation runs. `vendor` forces the
vendor-library path, `portable` forces the fallback/reference path, and
contradictory combinations with `NAVATALA_GPU_GEMM_VENDOR_MODE` fail loudly.
`mfma` selects the public F16-input/F32-output GEMM wrapper
(`navatala_gpu_gemm_f16_f32`) when the HIP transformer registry shard is built
into the runtime. The current Float32 ABI still returns
`NAVATALA_NOT_IMPLEMENTED` for `GEMM_IMPL=mfma` instead of silently dispatching
to an incompatible kernel.

Set `NAVATALA_GPU_ROCM_BENCH_INCLUDE_WRAPPER_GEMM=1` or pass
`--include-wrapper-gemm-benchmark` to add public-wrapper rows. The
`GEMM_F32_WRAPPER_VENDOR` rows call the public C ABI:

```text
navatala_gpu_gemm_f32
```

with:

```text
NAVATALA_GPU_GEMM_VENDOR_MODE=vendor
NAVATALA_GPU_GEMM_VENDOR_THRESHOLD_FLOPS=0
```

and compare the result/timing against direct `rocblas_sgemm`. The runner builds
the runtime with `GPU_RUNTIME_NAVATALA_FFI_STUB=OFF`, so the row fails loudly if
real runtime BLAS dispatch is unavailable instead of silently measuring the
stub/reference path. The row is distinct from `GEMM_F32`, which measures the
direct generated portable fallback kernel.

The same flag also adds `GEMM_F16_F32_WRAPPER_MFMA` rows. These call:

```text
navatala_gpu_gemm_f16_f32
```

with:

```text
NAVATALA_GPU_GEMM_IMPL=mfma
NAVATALA_GPU_GEMM_VENDOR_MODE=auto
NAVATALA_GPU_GEMM_MFMA_MODE=auto
```

and compare against `rocblas_gemm_ex` with F16 inputs and F32 output/compute.
Unlike the raw `GEMM_F16_MFMA_*` rows, this row exercises the user-facing
runtime wrapper and the CTA64/CTA128 dispatch policy. Its timing therefore
includes wrapper scalar-parameter setup and synchronization overhead. For the
`wrapper-semantics` matrix, the runner implicitly enables
`--include-mfma-benchmark` so the JSON also contains direct semantic full-tile
rows (`GEMM_F16_MFMA_CTA64_SHARED_SEMANTIC_ALPHA_BETA` and
`GEMM_F16_MFMA_CTA64_SHARED_SEMANTIC_TRANSPOSE`) next to the wrapper rows. Use
those direct rows to separate Phase 2 kernel cost from public-wrapper routing
cost.

For versioned ROCm installs on rented hosts, pass the installation root
explicitly or set `NAVATALA_GPU_ROCM_ROOT`:

```bash
./scripts/run_rocm_vendor_benchmarks.sh \
  --rocm-root /opt/rocm-7.2.4 \
  --quick \
  --iterations 5 \
  --include-wrapper-gemm-benchmark \
  --json /tmp/navatala_gpu_rocm_quick.json
```

The runner also honors `ROCM_PATH`/`HIP_PATH` and auto-detects `/opt/rocm*` when
no explicit root is provided.

Reports must not label the portable tiled GEMM result as vendor dispatch. The
manifest records raw kernel files as `portable_kernel` and public wrapper
dispatch paths under `libraryOperations[].implementation.kind =
vendor_dispatch`; the Markdown report should keep the same distinction visible.

## Real CSR Matrix Gate

Synthetic rowNnz matrices are useful for controlled tuning, but SpMV acceptance
for CFD requires at least one real pressure-matrix row distribution from
`openfoam_gpu`. A real-matrix fixture should publish:

- source case and git commit;
- rows, columns, nonzeros;
- row-nnz min, mean, p50, p90, p99, and max;
- index base and dtype;
- whether diagonal and boundary rows are included.

Large binary matrix fixtures should use an explicitly documented CSR binary
layout and should be kept out of the default no-GPU test path unless they are
small enough for normal source control.

## CI Runner

The generated public tree includes `.github/workflows/rocm-validation.yml`.
It expects a self-hosted runner with labels:

```text
self-hosted, linux, rocm, amd-gpu
```

The runner must have ROCm, HIP, rocBLAS, rocSPARSE, CMake, and a C++ compiler
installed. hipSPARSELt is optional; the workflow records whether it was
available.

This workflow is intentionally not mapped to GitHub-hosted CPU runners,
because a CPU-only skip would not prove HIP correctness.

Pull requests run quick mode on the self-hosted ROCm runner so correctness
feedback stays lightweight. Full benchmark runs should be triggered with
`workflow_dispatch` on a runner reserved for longer ROCm jobs.

## Evidence Rules

For a public release note or AMD forum follow-up, publish:

- the JSON report from `benchmarks/results/rocm_vendor_benchmark.json`;
- the generated Markdown report from
  `benchmarks/results/rocm_vendor_benchmark.md`;
- the workflow or shell logs, including `rocm_runtime_smoke.log`;
- GPU model, ROCm version, HIP compiler version, and driver/runtime details;
- generated/vendor back-to-back throughput mean per launch;
- generated/vendor runtime ratio;
- correctness result and maximum absolute error;
- whether hipSPARSELt was available and whether it was used.

Use `docs/benchmarks/ROCM_VALIDATION_TEMPLATE.md` for the final human-readable
report format.

## Non-Goals

This harness does not:

- prove correctness for every Navatala HIP kernel;
- claim performance parity with rocBLAS, rocSPARSE, or hipSPARSELt;
- replace future tuning work;
- benchmark CUDA, Vulkan, OpenCL, or Metal.

Those remain separate conformance and tuning tracks.
