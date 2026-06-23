# ROCm Validation Report Template

Use this template for public HIP/ROCm correctness and benchmark reports. A
completed report should be attached to the relevant release notes or linked
from `docs/NUMERICAL_CONFORMANCE.md`.

## Report Metadata

- Report id:
- Date:
- Author:
- Related tasks:
- Public package version:
- Public git tag/commit:
- Benchmark branch/commit, if different:

## Hardware And Software

| Field | Value |
| --- | --- |
| GPU model |  |
| GPU count |  |
| VRAM per GPU |  |
| CPU model |  |
| System memory |  |
| OS / kernel |  |
| ROCm version |  |
| HIP compiler version |  |
| Driver/runtime details |  |
| Python version, if used |  |
| CMake/Ninja versions |  |

## Build Configuration

```bash
# Record exact configure/build commands here.
cmake -S . -B build-rocm \
  -DNAVATALA_GPU_USE_HIP=ON \
  -DNAVATALA_GPU_USE_CUDA=OFF \
  -DNAVATALA_GPU_USE_VULKAN=OFF \
  -DNAVATALA_GPU_USE_OPENCL=OFF \
  -DNAVATALA_GPU_USE_METAL=OFF
cmake --build build-rocm -j
```

Environment:

```text
HIP_VISIBLE_DEVICES=
GPU_RUNTIME_BACKEND=hip
GPU_HIP_ARCH=
ROCM_PATH=
LD_LIBRARY_PATH=
```

## Operation Coverage

| Operation / kernel | Domain | Dtype | Shape(s) | Backend path | Status |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  | portable kernel / vendor dispatch | pass / fail / skipped |

Status meanings:

- `pass`: correctness and benchmark ran successfully;
- `fail`: correctness or runtime failed; include logs;
- `skipped`: not generated, unsupported dtype, missing vendor library, or not in
  this report's scope.

## Correctness Method

For each operation family, record:

- reference implementation;
- input data generation method and random seed;
- input sizes;
- absolute tolerance;
- relative tolerance;
- deterministic/reduction mode;
- number of repetitions;
- pass/fail summary;
- worst observed error and location.

## Benchmark Method

For each operation family, record:

- warm-up iterations;
- measured iterations;
- synchronization points;
- timing source;
- bytes moved / FLOPs formula;
- baseline implementation:
  - rocBLAS;
  - rocSPARSE;
  - hipSPARSELt;
  - CPU reference;
  - prior portable kernel baseline;
- median / p90 / min / max runtime;
- throughput and bandwidth where meaningful.

## Results Table

| Operation | Shape | Baseline | Navatala path | Correctness | Median runtime | Speedup | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  | pass/fail |  |  |  |

## Profiling Notes

Include `rocprof`, occupancy, memory bandwidth, kernel launch count, and any
observed synchronization overhead.

## Known Limitations

- Missing dtype/backend support:
- Numerical drift notes:
- Vendor-library gaps:
- Driver/runtime issues:
- Follow-up tasks:

## Attachments

- Raw logs:
- CSV/JSON benchmark output:
- Generated Markdown benchmark report:
- Plots:
- Build cache or command transcript:

## Sign-Off

- Reviewer:
- Date:
- Decision:
  - suitable for alpha note;
  - needs rerun;
  - blocks ROCm claim;
  - deferred.
