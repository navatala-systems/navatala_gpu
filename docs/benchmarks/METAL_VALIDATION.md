# Metal Validation And Runtime Tuning

This page defines the Apple Silicon validation pass for the Metal backend.
Metal source coverage is present for operations that the backend can represent,
but release-quality Metal claims require a macOS hardware run and a dated
artifact, just like the ROCm benchmark fixtures.

## Scope

The first Metal validation pass should answer three questions:

- do representative generated Metal kernels compile and produce numerically
  correct results through the public runtime;
- how much host-side overhead comes from command-buffer creation, queue
  synchronization, staging copies, and host-visible copies;
- whether opt-in command-buffer batching improves workloads with many small
  consecutive kernel submits.
- whether device-to-device copies stay on the Metal command stream instead of
  falling back to synchronized CPU-side shared-memory copies.

The pass is intentionally separate from ROCm benchmarking. Apple GPU memory,
compiler, command queue, and library behaviour differ enough that Metal needs
its own fixture and thresholds.

## Runtime Flags

Use these flags for the first tuning pass:

```bash
GPU_RUNTIME_BACKEND=metal
NAVATALA_GPU_RUNTIME_PROFILE=1
```

`NAVATALA_GPU_RUNTIME_PROFILE=1` prints one line per profile label at process
exit. The Metal backend reports:

- `submit`, `sync`
- `command_buffer`, `compute_encoder`, `blit_encoder`
- `h2d_count` / `h2d_bytes`
- `d2h_count` / `d2h_bytes`
- `d2d_count` / `d2d_bytes`
- `hosthost_count` / `hosthost_bytes`
- `host_visible_copy`
- `batch_dispatch`, `batch_flush`, `batch_limit_flush`
- `skipped_empty_sync`

The counters are diagnostic only and should not change runtime behaviour.

For the private device-buffer experiment, add:

```bash
NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS=1
```

This allocates `MemoryKind::Device` buffers with `MTLResourceStorageModePrivate`
instead of shared storage. Host transfers then use queued staging copies rather
than direct CPU access to shared memory.

## Opt-In Submit Batching

The Metal runtime has an opt-in compute-submit batching mode:

```bash
NAVATALA_GPU_METAL_BATCH_SUBMITS=1
NAVATALA_GPU_METAL_BATCH_LIMIT=64
```

When enabled, consecutive compute dispatches on the same queue are encoded into
one pending compute command buffer. The pending buffer is flushed before memory
copies, queue synchronization, event record/wait calls, queue destruction, or
when `NAVATALA_GPU_METAL_BATCH_LIMIT` dispatches have been encoded.

Default behaviour remains one command buffer per submit. Keep batching
default-off until Apple Silicon validation shows correctness and a clear
wall-time reduction for command-buffer-bound workloads.

## Suggested Manual Run

On the Mac validation host, prefer the packaged runner:

```bash
./scripts/run_metal_validation.sh \
  --result-dir benchmarks/results/metal-validation
```

The runner builds a Metal-only package, runs three passes, validates the JSON
artifact, and renders a Markdown report:

- `baseline`: normal one-command-buffer-per-submit runtime;
- `batched`: `NAVATALA_GPU_METAL_BATCH_SUBMITS=1`;
- `private`: `NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS=1`.

It writes:

```text
benchmarks/results/metal-validation/metal_validation.json
benchmarks/results/metal-validation/metal_validation.md
benchmarks/results/metal-validation/baseline.log
benchmarks/results/metal-validation/batched.log
benchmarks/results/metal-validation/private.log
```

The JSON contract can be checked on any machine:

```bash
python3 scripts/validate_metal_validation_json.py \
  benchmarks/results/metal-validation/metal_validation.json \
  --require-full
python3 scripts/render_metal_validation_report.py \
  benchmarks/results/metal-validation/metal_validation.json \
  --require-full \
  --output benchmarks/results/metal-validation/metal_validation.md
```

The repository also contains a CPU-checkable sample fixture:

```bash
python3 scripts/validate_metal_validation_json.py \
  benchmarks/fixtures/metal_validation.sample.json \
  --require-full
```

If the runner needs to be reproduced manually, use:

```bash
cmake -S . -B build-metal \
  -DNAVATALA_GPU_USE_METAL=ON \
  -DNAVATALA_GPU_USE_CUDA=OFF \
  -DNAVATALA_GPU_USE_HIP=OFF \
  -DNAVATALA_GPU_USE_VULKAN=OFF \
  -DNAVATALA_GPU_USE_OPENCL=OFF
cmake --build build-metal -j

GPU_RUNTIME_BACKEND=metal \
NAVATALA_GPU_RUNTIME_PROFILE=1 \
ctest --test-dir build-metal --output-on-failure
```

Then rerun the same workload with:

```bash
GPU_RUNTIME_BACKEND=metal \
NAVATALA_GPU_RUNTIME_PROFILE=1 \
NAVATALA_GPU_METAL_BATCH_SUBMITS=1 \
NAVATALA_GPU_METAL_BATCH_LIMIT=64 \
ctest --test-dir build-metal --output-on-failure
```

The public package CMake switch is `NAVATALA_GPU_USE_METAL`; use the
`NAVATALA_GPU_USE_*` option family for all new validation commands.

For any release claim, preserve:

- command used and environment flags;
- macOS version and Xcode Command Line Tools version;
- Apple GPU family / device name;
- commit hash;
- runtime profile output;
- correctness result;
- timing summary before and after batching.

## Acceptance Criteria

Submit batching can be promoted from experimental only if:

- all Metal correctness tests pass with and without batching;
- runtime profile output shows fewer command buffers for the same submit count;
- wall time improves for command-buffer-bound workloads;
- no workload regresses materially due to delayed command submission;
- documentation and backend coverage tables clearly distinguish validated Metal
  behaviour from unvalidated source coverage.

The first accepted artifact should be committed under a dated directory, for
example:

```text
benchmarks/fixtures/hardware_runs/YYYYMMDD_apple_metal_validation/
```

and should include `metal_validation.json`, `metal_validation.md`, and the
three pass logs. The sample fixture is only a schema example and must not be
quoted as hardware evidence.

The device-to-device blit path should be treated as correct if:

- copy-heavy tests produce the same results as the pre-change runtime;
- `d2d_count` increases for device-to-device transfers;
- `host_visible_copy` does not increase for device-to-device transfers;
- no additional queue synchronization is needed before later kernels on the
  same queue consume the copied data.

The direct host-visible copy path should be checked with both empty and dirty
queues. Empty queues should report `skipped_empty_sync` for direct H2D/D2H
copies that do not need a Metal fence. Dirty queues should synchronize before
CPU access so host-visible reads/writes do not race with earlier GPU work.

## Private-Buffer Validation

Keep private Metal device buffers default-off until they pass the Mac hardware
run. The validation pass must include:

- plain H2D, D2H, and D2D copies;
- offset H2D, D2H, and D2D copies;
- kernel execution with private input/output buffers;
- public C ABI copy tests with and without an explicit queue;
- runtime profile evidence that private-buffer transfers use queued copies
  rather than direct `host_visible_copy` paths.
