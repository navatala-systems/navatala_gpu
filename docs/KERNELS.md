# Kernels

`kernels/` contains the compute kernel corpus, with multiple backend
translations of each kernel. The corpus is **not** uniformly shaped across
backends; CUDA and HIP have the most complete coverage, OpenCL and Vulkan
are close behind, and Metal is significantly smaller.

See [`BACKEND_COVERAGE.md`](BACKEND_COVERAGE.md) for the current matrix
and `../kernels/manifest.json` for the machine-readable source of truth.

Why the gaps exist (measured against the CUDA reference):

- **HIP** is a 1:1 source-level analogue of CUDA — same programming
  model and intrinsics — and ships at full parity.

- **Vulkan** misses 22 kernels. The mix:
  - 5 BF16 / FP16 conversion and reduction kernels — Vulkan's `bfloat16`
    support arrived via extension recently and these kernels haven't been
    translated yet.
  - 5 stateful random-number kernels (`uniformLcg*`, `normalBoxMuller*`,
    `initRngState`) — Vulkan compute shaders lack the host-seeded
    persistent state these expect.
  - 7 F64-only kernels — Vulkan double-precision support varies by GPU
    and isn't universal.
  - 5 specialty kernels (`kaczmarzSweep`, `gamgProcAdd`, genetic
    `mutatePopulation`, `computeClusterSizes`) that haven't been ported.

- **OpenCL** misses 46 kernels, weighted toward two categories:
  - 23 LinearSolver kernels (mostly mixed-precision or atomic-dependent
    AMG / PCG variants where OpenCL's atomic and warp-vote semantics
    don't map cleanly).
  - 6 BF16 / FP16 cast and reduction kernels.
  - 13 are F64-only (less of a hard limit on OpenCL — many drivers do
    support `cl_khr_fp64` — but several of these depend on F64 *and*
    atomic operations together).
  - The rest are graph-generation utilities and a handful of AMG kernels.

- **Metal** misses 558 kernels — by far the largest gap, **but the cause
  is overwhelmingly simple**: Apple GPUs have effectively no
  double-precision support. ~96% of the missing kernels (535 F64-named +
  ~8 implicitly-F64 helpers) are dropped for this single reason. The
  remaining ~15 are BF16 casts and reductions, a handful of advanced
  PoissonPCG / MultigridDIC variants, and the F32-only specialty
  kernels (Kaczmarz, GAMG mixed-precision, deterministic scatter) that
  share Metal's porting backlog with Vulkan and OpenCL.

  The takeaway: Metal's "smaller" corpus isn't a quality issue with the
  Metal translation — it reflects Apple's product choice to skip
  double-precision GPU compute. Any kernel in the corpus that works in
  single precision has a Metal version; any kernel that requires F64
  doesn't.

If a kernel you need is missing from a particular backend, the kernel
files themselves are a useful starting point for porting — the
parameter signatures and bodies are simple enough that a domain expert
can usually translate one in an afternoon. PRs welcome.

## Domains covered

Approximate kernel counts by category (measured on the CUDA backend; exact
public counts are generated into [`KERNEL_INDEX.md`](KERNEL_INDEX.md)):

| Category                            | Kernels (CUDA) |
|-------------------------------------|---------------:|
| Volume-of-Fluid finite-volume CFD    |          ~140  |
| Algebraic multigrid (AMG)           |          ~240  |
| Classical iterative solvers         |           ~60  |
| BLAS / sparse linear algebra        |          ~100  |
| Neural / spectral operators         |           ~12  |
| Cross-platform ML library           |          ~790  |

### Volume-of-Fluid CFD primitives

Finite-volume operators reused across pressure-velocity solvers:
gradient assembly, Laplacian discretizations (uncorrected and corrected),
divergence operators (`divRhoPhiU`, etc.), surface-tension terms,
boundary-condition stencils, k-omega-SST turbulence model pieces, VOF
phase transport, and the halo / MPI-style communication primitives needed
to assemble these into a real solver.

### AMG and iterative solvers

Aggregation-based AMG (matrix aggregation, smoothing, restriction,
prolongation), classical Ruge–Stüben coarsening, matrix coloring, common
smoothers (Jacobi variants, Gauss–Seidel, Chebyshev, ILU(0)). Iterative
solvers include preconditioned CG, BiCGSTAB, IDR, GMRES, and a pipelined
ARIMA helper.

### BLAS and sparse linear algebra

SpMV in several sparse formats, sparse-dense conversions, reductions
(sum, max, dot product, axpy), matrix transpositions, and a handful of
dense linear-algebra routines.

### Neural and spectral operators

A small but useful set of operators for physics-informed neural networks
and Fourier neural operators (spectral convolutions, attention layers,
tensor product helpers).

### Generic ML library

A broader corpus of standard ML kernels (k-means, KNN, decision trees,
linear regression, statistical aggregations, t-digest, graph algorithms).
These overlap with what cuML provides on NVIDIA hardware but are
cross-platform here.

## Kernel naming convention

Kernel filenames and `extern "C"` entry points follow a uniform shape:

```
navatala_<module>_<op>_<dtype>
```

Examples: `navatala_linalg_lanczos_step_f32`, `navatala_sparse_amg_smoother_f64`,
`navatala_dataframe_top_k_select_i32`. The `<module>` segment is one of the
public modules (`linalg`, `sparse`, `dataframe`, `graph`, `ml`, `cfd`,
`runtime`, `kernels`). The `<op>` segment is the operation name in
snake_case. The `<dtype>` segment is a lowercase type tag — `f16`, `f32`,
`f64`, `bf16`, `i8`/`i16`/`i32`/`i64`, `u8`/`u16`/`u32`/`u64`, `c32`/`c64`
— and is omitted when the kernel has no type parameter.

These names:

- are **stable** — once a kernel ships under a given name it will not be
  renamed without a major-version bump, so consumers can hard-code them
  in registries and dispatch tables;
- are unique across the entire corpus (no two kernels share a name);
- map onto the source-tree layout: `kernels/<backend>/<module>/<op>_<dtype>.<ext>`
  matches the symbol body.

If you want even shorter names for application code, wrap the entry points
in a dispatch shim of your own — the runtime doesn't require you to use
the verbatim entry-point string for anything except the call to
`Program::createProgram()`.

## How to read a kernel

Each kernel file follows a consistent shape:

```cuda
extern "C" __global__ void axpyCells(
    const int n,
    const float* __restrict__ paramsF,
    const float* __restrict__ x,
    float* __restrict__ y)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float yi = y[i];
    y[i] = (yi + (paramsF[0] * x[i]));
}
```

Conventions:

- Every kernel is `extern "C"` so it can be looked up by name from the
  runtime registry.
- Arguments are typed primitive pointers and integer sizes; scalars are
  passed via a `paramsF` (float) or `paramsI` (int) array, not by value.
- Bounds checks are explicit (`if (i >= n) return;`) — the host launches
  with grid sizes rounded up.
- Floating-point literals occasionally appear as bit-pattern reinterprets
  (`__uint_as_float(0x3f4ccccdu)` is `0.8f`); this is a translation
  artifact, not a stylistic choice.

OpenCL, Vulkan GLSL, and Metal variants use the same parameter ordering
and the same per-thread index calculation, with the obvious backend-specific
intrinsic substitutions (`get_global_id(0)`, `gl_GlobalInvocationID.x`,
`__gid.x`).

## Manifest metadata

`kernels/manifest.json` is the machine-readable index of the kernel corpus
shipped in this package. It intentionally omits kernel bodies and program
graphs, but retains the public information a runtime user needs:

- formal/source contract name and stable entry point;
- domain/category;
- argument names, compact types, access modes, and memory spaces;
- workgroup size and default dispatch;
- determinism and capability requirements;
- backend availability and source/binary file paths.

The manifest drives generated docs such as [`BACKEND_COVERAGE.md`](BACKEND_COVERAGE.md)
and [`KERNEL_INDEX.md`](KERNEL_INDEX.md), and is the intended input for
future wrapper/API generation.

## Current limitations

- **Sparse per-kernel prose**: per-kernel docstrings are intentionally
  minimal. The manifest provides structured metadata; full domain
  explanations live in higher-level docs and references.
- **Limited backend-specific tuning**: each backend currently gets the
  same translation. This trades absolute peak performance for portability
  and cross-backend consistency.
- **No tests bundled in `kernels/`**: the runtime tests under
  `runtime/tests/` cover the loader and execution mechanics; numerical
  correctness of individual kernels is validated upstream against
  reference implementations.

## Reusing individual kernels

You can lift a single kernel file out of this distribution and use it
standalone. The host-side kernel registry (header at
`runtime/include/navatala/_registry.hpp`, source under
`runtime/src/internal/`) is convenient but not required — any reasonable
JIT pipeline that takes a source string will work.

If you find a numerical bug, please file an issue with the kernel file
name and a minimal reproducer; fixes land in all five backends together
to keep behaviour consistent across vendors.
