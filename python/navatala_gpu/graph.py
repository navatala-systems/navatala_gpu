# SPDX-License-Identifier: Apache-2.0
from ._core import call_extension
from .runtime import supports as _runtime_supports, validate_call

__navatala_module__ = "graph"
_BINDINGS = ("out_degree", "in_degree")
__all__ = ("out_degree", "in_degree", "list_bindings", "supports")

def list_bindings() -> tuple[str, ...]:
    """Return the public operation names exposed by this module."""
    return _BINDINGS

def supports(operation: str, *, backend: str | None = None, dtype: str | None = None) -> bool:
    """Return whether this module exposes an operation for the requested backend/dtype."""
    if operation not in _BINDINGS:
        return False
    return _runtime_supports(f"{__navatala_module__}.{operation}", backend=backend, dtype=dtype)

def out_degree(row_offsets, degrees, vertex_count, backend="auto", *, queue=None, sync=False):
    """
    Compute per-vertex out-degree from CSR row offsets.

    The graph descriptor uses uint32 row_offsets with length vertex_count + 1. The output degrees tensor has length vertex_count.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: uint32
    - hip: uint32
    - opencl: uint32
    - vulkan: uint32
    - metal: uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("graph", "out_degree", {"row_offsets": row_offsets, "degrees": degrees}, {"vertex_count": vertex_count, "backend": backend})
    return call_extension("graph", "out_degree", row_offsets, degrees, vertex_count, backend, queue=queue, sync=sync)

def in_degree(col_offsets, degrees, vertex_count, backend="auto", *, queue=None, sync=False):
    """
    Compute per-vertex in-degree from CSC column offsets.

    The graph descriptor uses uint32 CSC/transposed-CSR col_offsets with length vertex_count + 1. The output degrees tensor has length vertex_count.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: uint32
    - hip: uint32
    - opencl: uint32
    - vulkan: uint32
    - metal: uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("graph", "in_degree", {"col_offsets": col_offsets, "degrees": degrees}, {"vertex_count": vertex_count, "backend": backend})
    return call_extension("graph", "in_degree", col_offsets, degrees, vertex_count, backend, queue=queue, sync=sync)