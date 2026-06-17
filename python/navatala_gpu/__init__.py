# SPDX-License-Identifier: Apache-2.0
"""navatala_gpu — multi-backend GPU operations with Python bindings.

Public surface:
    ``__version__`` — package SemVer string (matches ``pyproject.toml``).
    ``__abi_version__`` — integer C ABI revision; bumps on each break.

Internal modules (underscore-prefixed) are not part of the public API.
"""
__version__ = "0.1.1"
__abi_version__ = 6
__navatala_ffi_abi_version__ = __abi_version__  # back-compat alias; prefer ``__abi_version__``
from . import _meta as _meta  # noqa: F401  internal metadata; not in __all__
from .runtime import (
    Device, Queue, Event, Buffer, Tensor, DLPackAdapter,
    get_capabilities, supports, can_zero_copy_dlpack,
    dlpack_backend_device_type, dlpack_device_to_backend,
)
from ._exceptions import (
    NavatalaError, InvalidHandleError, GpuRuntimeError, OutOfMemoryError,
    TimeoutGpuError, CancelledError, AlreadyExistsError, NotFoundError,
    PermissionDeniedError, NavatalaIOError, NavatalaInternalError, InvalidShapeError,
    InvalidDTypeError, UnsupportedBackendError, NonContiguousError,
    InvalidArgumentError, RuntimeFailureError, MissingExtensionError,
)

__all__ = ["Device", "Queue", "Event", "Buffer", "Tensor", "DLPackAdapter", "get_capabilities", "supports", "can_zero_copy_dlpack", "dlpack_backend_device_type", "dlpack_device_to_backend", "NavatalaError", "InvalidHandleError", "GpuRuntimeError", "OutOfMemoryError", "TimeoutGpuError", "CancelledError", "AlreadyExistsError", "NotFoundError", "PermissionDeniedError", "NavatalaIOError", "NavatalaInternalError", "InvalidShapeError", "InvalidDTypeError", "UnsupportedBackendError", "NonContiguousError", "InvalidArgumentError", "RuntimeFailureError", "MissingExtensionError", "linalg", "sparse", "dataframe", "graph", "ml", "cfd", "kernels"]
