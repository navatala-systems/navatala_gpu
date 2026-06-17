# SPDX-License-Identifier: Apache-2.0
from ._core import call_extension
from .runtime import supports as _runtime_supports, validate_call

__navatala_module__ = "linalg"
_BINDINGS = ("axpy", "axpy_dispatch", "gemm", "nrm2")
__all__ = ("axpy", "axpy_dispatch", "gemm", "nrm2", "list_bindings", "supports")

def list_bindings() -> tuple[str, ...]:
    """Return the public operation names exposed by this module."""
    return _BINDINGS

def supports(operation: str, *, backend: str | None = None, dtype: str | None = None) -> bool:
    """Return whether this module exposes an operation for the requested backend/dtype."""
    if operation not in _BINDINGS:
        return False
    return _runtime_supports(f"{__navatala_module__}.{operation}", backend=backend, dtype=dtype)

def axpy(x, y, alpha=1.000000, backend="auto", *, queue=None, sync=False):
    """
    Compute y = alpha * x + y on the selected GPU backend.

    The Python binding accepts DLPack-compatible tensors and returns the updated y tensor.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32
    - hip: float32
    - opencl: float32
    - vulkan: float32
    - metal: float32

    Synchronization:
    The binding returns after enqueue unless the caller requests synchronization.
    """
    validate_call("linalg", "axpy", {"x": x, "y": y}, {"alpha": alpha, "backend": backend})
    return call_extension("linalg", "axpy", x, y, alpha, backend, queue=queue, sync=sync)

def axpy_dispatch(x, y, alpha=1.000000, backend="auto", *, queue=None, sync=False):
    """
    Compute y = alpha * x + y through a dtype-dispatched binding group.

    The Python binding validates dispatch-group dtype routing before invoking the concrete AXPY implementation.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32
    - hip: float32
    - opencl: float32
    - vulkan: float32
    - metal: float32

    Synchronization:
    The binding returns after enqueue unless the caller requests synchronization.
    """
    validate_call("linalg", "axpy_dispatch", {"x": x, "y": y}, {"alpha": alpha, "backend": backend})
    return call_extension("linalg", "axpy_dispatch", x, y, alpha, backend, queue=queue, sync=sync)

def gemm(a, b, c, alpha=1.000000, beta=0.000000, backend="auto", *, queue=None, sync=False):
    """
    Compute C = alpha * A @ B + beta * C on the selected GPU backend.

    The Python binding validates rank-2 tensors and backend dtype support before dispatch.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32
    - hip: float32
    - opencl: float32
    - vulkan: float32
    - metal: float32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("linalg", "gemm", {"a": a, "b": b, "c": c}, {"alpha": alpha, "beta": beta, "backend": backend})
    return call_extension("linalg", "gemm", a, b, c, alpha, beta, backend, queue=queue, sync=sync)

def nrm2(x, result, backend="auto", *, queue=None, sync=False):
    """
    Compute result[0] = ||x||_2 on the selected GPU backend.

    The result argument is an explicit scalar float32 Tensor, keeping output allocation visible to the caller.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32
    - hip: float32
    - opencl: float32
    - vulkan: float32
    - metal: float32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("linalg", "nrm2", {"x": x, "result": result}, {"backend": backend})
    return call_extension("linalg", "nrm2", x, result, backend, queue=queue, sync=sync)