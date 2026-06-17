# runtime/

Cross-platform GPU runtime library.

Public API is declared in [`include/gpu_runtime.h`](include/gpu_runtime.h);
backend implementations live in `src/backend_*.cpp`; tests live in
`tests/`.

For the project-wide README, build instructions, and architecture overview,
see the [top-level README](../README.md) and [docs/](../docs/).

```bash
# Build just the runtime (from the navatala_gpu/ root):
cmake -S . -B build -DNAVATALA_GPU_BUILD_RUNTIME=ON
cmake --build build -j

# Run the tests:
ctest --test-dir build --output-on-failure
```
