# SPDX-License-Identifier: Apache-2.0
from ._core import call_extension
from .runtime import supports as _runtime_supports, validate_call

__navatala_module__ = "ml"
_BINDINGS = ("mean_squared_error", "root_mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "r2_score", "explained_variance_score")
__all__ = ("mean_squared_error", "root_mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "r2_score", "explained_variance_score", "list_bindings", "supports")

def list_bindings() -> tuple[str, ...]:
    """Return the public operation names exposed by this module."""
    return _BINDINGS

def supports(operation: str, *, backend: str | None = None, dtype: str | None = None) -> bool:
    """Return whether this module exposes an operation for the requested backend/dtype."""
    if operation not in _BINDINGS:
        return False
    return _runtime_supports(f"{__navatala_module__}.{operation}", backend=backend, dtype=dtype)

def mean_squared_error(y_true, y_pred, result, count, backend="auto", *, queue=None, sync=False):
    """
    Compute float32 mean squared error.

    y_true and y_pred are rank-1 float32 tensors with length count. result is an explicit scalar float32 tensor.

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
    validate_call("ml", "mean_squared_error", {"y_true": y_true, "y_pred": y_pred, "result": result}, {"count": count, "backend": backend})
    return call_extension("ml", "mean_squared_error", y_true, y_pred, result, count, backend, queue=queue, sync=sync)

def root_mean_squared_error(y_true, y_pred, result, count, backend="auto", *, queue=None, sync=False):
    """
    Compute float32 root mean squared error.

    y_true and y_pred are rank-1 float32 tensors with length count. result is an explicit scalar float32 tensor.

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
    validate_call("ml", "root_mean_squared_error", {"y_true": y_true, "y_pred": y_pred, "result": result}, {"count": count, "backend": backend})
    return call_extension("ml", "root_mean_squared_error", y_true, y_pred, result, count, backend, queue=queue, sync=sync)

def mean_absolute_error(y_true, y_pred, result, count, backend="auto", *, queue=None, sync=False):
    """
    Compute float32 mean absolute error.

    y_true and y_pred are rank-1 float32 tensors with length count. result is an explicit scalar float32 tensor.

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
    validate_call("ml", "mean_absolute_error", {"y_true": y_true, "y_pred": y_pred, "result": result}, {"count": count, "backend": backend})
    return call_extension("ml", "mean_absolute_error", y_true, y_pred, result, count, backend, queue=queue, sync=sync)

def mean_absolute_percentage_error(y_true, y_pred, result, count, backend="auto", *, queue=None, sync=False):
    """
    Compute float32 mean absolute percentage error.

    y_true and y_pred are rank-1 float32 tensors with length count. result is an explicit scalar float32 tensor. y_true == 0 follows IEEE-754 division semantics.

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
    validate_call("ml", "mean_absolute_percentage_error", {"y_true": y_true, "y_pred": y_pred, "result": result}, {"count": count, "backend": backend})
    return call_extension("ml", "mean_absolute_percentage_error", y_true, y_pred, result, count, backend, queue=queue, sync=sync)

def r2_score(y_true, y_pred, y_mean, result, count, backend="auto", *, queue=None, sync=False):
    """
    Compute float32 R2 score.

    y_true and y_pred are rank-1 float32 tensors with length count. y_mean is an explicit scalar float32 tensor containing the precomputed mean of y_true; result is an explicit scalar float32 tensor. Constant y_true inputs follow IEEE-754 division semantics.

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
    validate_call("ml", "r2_score", {"y_true": y_true, "y_pred": y_pred, "y_mean": y_mean, "result": result}, {"count": count, "backend": backend})
    return call_extension("ml", "r2_score", y_true, y_pred, y_mean, result, count, backend, queue=queue, sync=sync)

def explained_variance_score(y_true, y_pred, mean_residuals, mean_true, result, count, backend="auto", *, queue=None, sync=False):
    """
    Compute float32 explained variance score.

    y_true and y_pred are rank-1 float32 tensors with length count. mean_residuals and mean_true are explicit scalar float32 tensors; result is an explicit scalar float32 tensor. Constant y_true inputs follow IEEE-754 division semantics.

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
    validate_call("ml", "explained_variance_score", {"y_true": y_true, "y_pred": y_pred, "mean_residuals": mean_residuals, "mean_true": mean_true, "result": result}, {"count": count, "backend": backend})
    return call_extension("ml", "explained_variance_score", y_true, y_pred, mean_residuals, mean_true, result, count, backend, queue=queue, sync=sync)