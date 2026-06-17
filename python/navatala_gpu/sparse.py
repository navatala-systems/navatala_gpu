# SPDX-License-Identifier: Apache-2.0
from ._core import call_extension
from .runtime import InvalidArgumentError, Tensor, supports as _runtime_supports, validate_call

__navatala_module__ = "sparse"
_BINDINGS = ("csr_spmv",)
__all__ = ("csr_spmv", "CSRMatrix", "list_bindings", "supports")

def list_bindings() -> tuple[str, ...]:
    """Return the public operation names exposed by this module."""
    return _BINDINGS

def supports(operation: str, *, backend: str | None = None, dtype: str | None = None) -> bool:
    """Return whether this module exposes an operation for the requested backend/dtype."""
    if operation not in _BINDINGS:
        return False
    return _runtime_supports(f"{__navatala_module__}.{operation}", backend=backend, dtype=dtype)

class CSRMatrix:
    """CSR sparse matrix descriptor backed by navatala_gpu Tensor objects."""

    def __init__(self, row_offsets, col_indices, values, *, shape):
        if len(tuple(shape)) != 2:
            raise InvalidArgumentError("CSRMatrix shape must be a (rows, cols) pair")
        rows, cols = (int(shape[0]), int(shape[1]))
        if rows < 0 or cols < 0:
            raise InvalidArgumentError("CSRMatrix shape dimensions must be non-negative")
        self.row_offsets = row_offsets
        self.col_indices = col_indices
        self.values = values
        self.shape = (rows, cols)
        self.rows = rows
        self.cols = cols
        self.nnz = int(getattr(values, "shape", (0,))[0])
        self._validate()

    @classmethod
    def from_numpy(cls, row_offsets, col_indices, values, *, shape, device=None, queue=None):
        """Create a CSRMatrix from NumPy-compatible host arrays using explicit copies."""
        import numpy as np
        row_offsets_t = Tensor.from_numpy(np.asarray(row_offsets, dtype=np.uint32), device=device, queue=queue)
        col_indices_t = Tensor.from_numpy(np.asarray(col_indices, dtype=np.uint32), device=device, queue=queue)
        values_t = Tensor.from_numpy(np.asarray(values, dtype=np.float32), device=device, queue=queue)
        return cls(row_offsets_t, col_indices_t, values_t, shape=shape)

    @property
    def backend(self) -> str:
        return getattr(getattr(self.values, "device", None), "backend", "auto")

    @property
    def dtype(self) -> str:
        return str(getattr(self.values, "dtype", "float32"))

    def _validate(self) -> None:
        row_shape = tuple(getattr(self.row_offsets, "shape", ()))
        col_shape = tuple(getattr(self.col_indices, "shape", ()))
        val_shape = tuple(getattr(self.values, "shape", ()))
        if row_shape != (self.rows + 1,):
            raise InvalidArgumentError(f"CSR row_offsets shape must be ({self.rows + 1},), got {row_shape}")
        if len(col_shape) != 1 or len(val_shape) != 1 or col_shape != val_shape:
            raise InvalidArgumentError("CSR col_indices and values must be rank-1 tensors with identical shape")
        if str(getattr(self.row_offsets, "dtype", "")) != "uint32" or str(getattr(self.col_indices, "dtype", "")) != "uint32":
            raise InvalidArgumentError("CSR row_offsets and col_indices must have dtype uint32")
        if str(getattr(self.values, "dtype", "")) != "float32":
            raise InvalidArgumentError("CSR values must have dtype float32")
        devices = {getattr(getattr(t, "device", None), "backend", None) for t in (self.row_offsets, self.col_indices, self.values)}
        if len(devices) != 1:
            raise InvalidArgumentError("CSR descriptor tensors must live on the same backend")
        self.nnz = int(val_shape[0])

    def matvec(self, x, out=None, *, backend="auto", queue=None, sync=False):
        """Compute y = A * x using the csr_spmv binding."""
        if out is None:
            out = Tensor.empty((self.rows,), dtype="float32", device=getattr(self.values, "device", None), queue=queue)
        return csr_spmv(
            self.row_offsets, self.col_indices, self.values, x, out,
            rows=self.rows, cols=self.cols, nnz=self.nnz, backend=backend,
            queue=queue, sync=sync,
        )

def csr_spmv(row_offsets, col_indices, values, x, y, rows, cols, nnz, backend="auto", *, queue=None, sync=False):
    """
    Compute y = A * x for a float32 CSR sparse matrix.

    The CSR descriptor uses uint32 row_offsets/col_indices and float32 values. rows, cols, and nnz make the descriptor contract explicit at the Python boundary.

    Maturity: experimental

    Supported backend/dtype pairs:
    - cuda: float32, uint32
    - hip: float32, uint32
    - opencl: float32, uint32
    - vulkan: float32, uint32
    - metal: float32, uint32

    Synchronization:
    Host synchronization is never inserted by default.
    """
    validate_call("sparse", "csr_spmv", {"row_offsets": row_offsets, "col_indices": col_indices, "values": values, "x": x, "y": y}, {"rows": rows, "cols": cols, "nnz": nnz, "backend": backend})
    return call_extension("sparse", "csr_spmv", row_offsets, col_indices, values, x, y, rows, cols, nnz, backend, queue=queue, sync=sync)