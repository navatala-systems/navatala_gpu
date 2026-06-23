# examples/

Worked examples that call the public C and C++ APIs. They are useful both as a
sanity check that an end-to-end build works on your machine and as a template
for wiring Navatala GPU into your own code.

## axpy_example.cpp

Computes `y = y + alpha * x` through `navatala_gpu_axpy_f32`, verifies the
result against the expected closed-form output, and prints PASS/FAIL. This
demonstrates backend/context setup through the stable C ABI.

## wrapper_axpy_example.cpp

Computes the same operation through `navatala::resources`,
`navatala::buffer`, and `navatala::linalg::axpy`. This is the preferred shape
for C++ application code: wrappers validate handles, data types, and shapes
before crossing into the C ABI.

## wrapper_spmv_example.cpp

Computes `y = A * x` for a small CSR sparse matrix through
`navatala::sparse::csr_spmv`. This demonstrates the stable sparse wrapper
surface with `uint32` CSR row/column indices and `float32` values.

## registry_lookup_example.cpp

Looks up a generated sample kernel source and ABI manifest from a host-registry
partition. It does not create a GPU context; it is a catalog/discoverability
smoke test for the generated kernel corpus. The full internal registry
aggregator also ships under `runtime/src/internal/`, but is intentionally not
linked into this default example because it depends on the optional library-op
dispatch layer.

```
cmake -S . -B build
cmake --build build -j
./build/examples/axpy_example
./build/examples/wrapper_axpy_example
./build/examples/wrapper_spmv_example
./build/examples/registry_lookup_example
```

The default build uses the stub FFI, so the example also passes on CPU-only CI.
Real backend builds can set `-DGPU_RUNTIME_NAVATALA_FFI_STUB=OFF` and select
available backends through the top-level `NAVATALA_GPU_USE_*` options.
