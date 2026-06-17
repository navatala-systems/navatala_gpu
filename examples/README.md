# examples/

Worked examples that call the public `navatala/navatala_ffi.h` ABI. They are
useful both as a sanity check that an end-to-end build works on your machine
and as a template for wiring Navatala GPU into your own code.

## axpy_example

Computes `y = y + alpha * x` through `navatala_gpu_axpy_f32`, verifies the
result against the expected closed-form output, and prints PASS/FAIL.

```
cmake -S . -B build
cmake --build build -j
./build/examples/axpy_example
```

The default build uses the stub FFI, so the example also passes on CPU-only CI.
Real backend builds can set `-DGPU_RUNTIME_NAVATALA_FFI_STUB=OFF` and select
available backends through the top-level `NAVATALA_GPU_USE_*` options.
