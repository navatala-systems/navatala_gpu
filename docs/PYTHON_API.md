# Navatala GPU Python API Guide

This document describes the intended public Python API for `navatala_gpu`. The
Python package is a curated interface over the generated GPU library and runtime;
it is not a raw dump of generated kernels.

The stable public API uses explicit tensor objects, explicit copy boundaries, and
capability checks. DLPack is a zero-copy exchange path only; incompatible
backend pairs raise clear errors and require an explicit copy API outside
DLPack.

## 1. Import And Capability Queries

`import navatala_gpu` must work even on hosts without GPU drivers. Backends are
loaded lazily and reported through capability queries.

```python
import navatala_gpu as ng

caps = ng.get_capabilities()
print(caps.available_backends)

if not ng.supports("linalg.axpy", backend="hip", dtype="float32"):
    raise RuntimeError("AXPY float32 is not available on HIP in this build")
```

## 2. Runtime Objects

Before allocating tensors or running algorithms, create a `Device` and a
`Queue`. Long-running operations may enqueue work and return before the queue is
complete; call `queue.sync()` when host-visible results are required.

```python
import navatala_gpu as ng

dev = ng.Device("hip", 0)
queue = dev.create_queue()
```

Resource-owning objects support explicit cleanup through `.close()` and context
manager usage. Finalizers are best-effort and no-throw; they are not the normal
error-reporting cleanup path.

```python
with ng.Device("hip", 0) as dev:
    queue = dev.create_queue()
    # allocate tensors and launch work here
```

## 3. NumPy Copy-In / Copy-Out

NumPy arrays are CPU arrays. Moving NumPy data to a GPU tensor is an explicit
host-to-device copy. Moving a GPU tensor back to NumPy is an explicit
device-to-host copy and synchronizes as needed.

```python
import numpy as np
import navatala_gpu as ng

dev = ng.Device("metal", 0)
queue = dev.create_queue()

x_np = np.ones((1000,), dtype=np.float32)
x = ng.Tensor.from_numpy(x_np, device=dev, queue=queue)
y = ng.empty_like(x)

ng.linalg.axpy(alpha=2.0, x=x, y=y, queue=queue)
queue.sync()

out_np = y.to_numpy()
```

## 4. DLPack Zero-Copy Interop

Stable generated operations consume and return `navatala_gpu.Tensor` objects.
External GPU tensors enter the runtime explicitly through
`Tensor.from_dlpack(obj)`.

```python
import torch
import navatala_gpu as ng

dev = ng.Device("cuda", 0)
queue = dev.create_queue()

x_torch = torch.randn(1024, 1024, device="cuda")
y_torch = torch.zeros_like(x_torch)

x = ng.Tensor.from_dlpack(x_torch, device=dev, queue=queue)
y = ng.Tensor.from_dlpack(y_torch, device=dev, queue=queue)

ng.linalg.axpy(alpha=2.0, x=x, y=y, queue=queue)
queue.sync()

# y_torch now observes the updated data when the backend pair supports
# same-backend zero-copy DLPack.
```

Cross-backend imports, for example PyTorch CUDA into Navatala HIP, raise a
`RuntimeError` with guidance to use an explicit copy path. DLPack never performs
implicit host copies.

## 5. Curated Modules

The public API is divided into logical modules containing high-level operations.

| Module | Maturity | Examples |
| --- | --- | --- |
| `ng.runtime` | Stable | `Device`, `Queue`, `Tensor`, `get_capabilities` |
| `ng.linalg` | Stable | `axpy`, `gemm`, `norm` |
| `ng.sparse` | Stable | `csr_matrix`, `spmv`, `spmm` |
| `ng.dataframe` | Stable | `Table`, `Column`, `join`, `groupby` |
| `ng.graph` | Stable | `Graph`, `pagerank`, `bfs` |
| `ng.ml` | Experimental | `KMeans`, `RandomForest` |
| `ng.cfd` | Experimental | `fno_forward`, `mgn_forward` |
| `ng.kernels` | Internal | raw generated launch helpers |

Raw generated kernels are internal by default. Advanced raw-kernel access, if
enabled, is not covered by the stable API contract.

## 6. Error Behavior

- Unsupported backend or dtype combinations raise `NotImplementedError`.
- Invalid shapes, ranks, strides, or scalar parameters raise `ValueError`.
- GPU runtime failures raise `navatala_gpu.GpuRuntimeError`.
- Cleanup errors are reported by `.close()` or context-manager exits.
- `__del__` / `tp_finalize` paths are no-throw and logging-only.

Generated docstrings and `.pyi` type stubs are produced from the formal Python
binding contracts, including backend support, dtype support, synchronization
policy, and maturity labels.
