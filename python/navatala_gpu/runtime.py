# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
from importlib import resources as importlib_resources
from dataclasses import dataclass, field
from typing import Any, Optional

from . import _core
from ._exceptions import (
    InvalidShapeError, InvalidDTypeError, UnsupportedBackendError,
    NonContiguousError, InvalidArgumentError, RuntimeFailureError,
    GpuRuntimeError,
)

_KNOWN_BACKENDS = ["cuda", "hip", "opencl", "vulkan", "metal"]

def _load_manifest() -> dict[str, Any]:
    try:
        text = importlib_resources.files(__package__).joinpath("api_manifest.json").read_text()
    except Exception:
        return {"abiVersion": 6, "bindings": []}
    return json.loads(text)

_API_MANIFEST = _load_manifest()
_LOG = logging.getLogger(__name__)

_DLPACK_DEVICE_TO_BACKEND = {
    2: "cuda",
    3: "cuda_host",
    4: "opencl",
    7: "vulkan",
    8: "metal",
    10: "hip",
    11: "hip_host",
    13: "cuda_managed",
}
_DLPACK_BACKEND_TO_DEVICE = {
    "cuda": 2,
    "hip": 10,
    "opencl": 4,
    "vulkan": 7,
    "metal": 8,
}
_ZERO_COPY_DLPACK_BACKENDS = {"cuda", "hip"}

_DTYPE_ALIASES = {
    "float16": {"float16", "Float16"},
    "float32": {"float32", "Float32"},
    "float64": {"float64", "Float64"},
    "bfloat16": {"bfloat16", "bf16", "BFloat16"},
    "bf16": {"bfloat16", "bf16", "BFloat16"},
    "int8": {"int8", "Int8"},
    "uint8": {"uint8", "UInt8"},
    "int16": {"int16", "Int16"},
    "uint16": {"uint16", "UInt16"},
    "int32": {"int32", "Int32"},
    "uint32": {"uint32", "UInt32"},
    "int64": {"int64", "Int64"},
    "uint64": {"uint64", "UInt64"},
}

def _dtype_supported(requested: str, supported: list[str]) -> bool:
    aliases = _DTYPE_ALIASES.get(str(requested), {str(requested)})
    return any(item in aliases for item in supported)

def dlpack_device_to_backend(device_type: int) -> str:
    try:
        return _DLPACK_DEVICE_TO_BACKEND[int(device_type)]
    except (KeyError, TypeError, ValueError) as exc:
        raise UnsupportedBackendError(f"Unsupported DLPack device type: {device_type}") from exc

def dlpack_backend_device_type(backend: str | Device) -> int:
    backend_name = _backend_name(backend)
    try:
        return _DLPACK_BACKEND_TO_DEVICE[backend_name]
    except KeyError as exc:
        raise UnsupportedBackendError(f"Backend {backend_name!r} has no DLPack device mapping") from exc

def _backend_name(value) -> str:
    if isinstance(value, Device):
        value = value.backend
    if isinstance(value, Tensor):
        value = value.device.backend
    if isinstance(value, tuple) and value:
        value = dlpack_device_to_backend(value[0])
    elif isinstance(value, int):
        value = dlpack_device_to_backend(value)
    backend = str(value).lower()
    return {"rocm": "hip"}.get(backend, backend)

def can_zero_copy_dlpack(producer, consumer=None, *, dtype: str | None = None) -> bool:
    """Return whether the runtime supports zero-copy DLPack exchange."""
    try:
        producer_backend = _backend_name(producer)
        consumer_backend = producer_backend if consumer is None else _backend_name(consumer)
    except UnsupportedBackendError:
        return False
    if dtype is not None and str(dtype) not in _DTYPE_ALIASES:
        return False
    return producer_backend == consumer_backend and producer_backend in _ZERO_COPY_DLPACK_BACKENDS

def _operation_support() -> dict[str, Any]:
    operations: dict[str, Any] = {}
    for binding in _API_MANIFEST.get("bindings", []):
        key = f"{binding.get('module')}.{binding.get('pythonName')}"
        operations[key] = {
            "maturity": binding.get("maturity"),
            "target": binding.get("target"),
            "implementation": binding.get("implementation", {"kind": "portable_kernel", "vendorBacked": False}),
            "syncPolicy": binding.get("syncPolicy"),
            "backends": {
                row.get("backend"): row.get("dtypes", [])
                for row in binding.get("backendSupport", [])
            },
        }
    return operations

def _exception_for_code(code: str):
    return {
        "invalid_shape": InvalidShapeError,
        "invalid_dtype": InvalidDTypeError,
        "unsupported_backend": UnsupportedBackendError,
        "non_contiguous": NonContiguousError,
        "invalid_argument": InvalidArgumentError,
        "runtime_failure": RuntimeFailureError,
    }.get(code, InvalidArgumentError)

def _binding_for(module: str, name: str) -> dict[str, Any] | None:
    for binding in _API_MANIFEST.get("bindings", []):
        if binding.get("module") == module and binding.get("pythonName") == name:
            return binding
    return None

def _tensor_shape(value) -> tuple:
    return tuple(getattr(value, "shape", ()))

def _tensor_dtype(value) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)

def _tensor_backend(value) -> str | None:
    device = getattr(value, "device", None)
    backend = getattr(device, "backend", None)
    if backend in {None, "", "auto"}:
        return None
    return str(backend)

def _scalar_backend(scalars: dict[str, Any]) -> str | None:
    value = scalars.get("backend")
    if value in {None, "", "auto"}:
        return None
    return str(value)

def _requested_backend(tensors: dict[str, Any], scalars: dict[str, Any]) -> str | None:
    backend = _scalar_backend(scalars)
    if backend is not None:
        return backend
    for tensor in tensors.values():
        backend = _tensor_backend(tensor)
        if backend is not None:
            return backend
    return None

def _check_dtype_constraint(name: str, value, spec: dict[str, Any]) -> None:
    dtype = _tensor_dtype(value)
    if dtype is None:
        return
    kind = spec.get("kind", "any")
    if kind == "fixed" and not _dtype_supported(dtype, [spec.get("dtype")]):
        raise InvalidDTypeError(f"{name} must have dtype {spec.get('dtype')}, got {dtype}")
    if kind == "one_of" and not _dtype_supported(dtype, spec.get("dtypes", [])):
        raise InvalidDTypeError(f"{name} has unsupported dtype {dtype}")

def _check_shape_constraint(name: str, value, spec: dict[str, Any], tensors: dict[str, Any], symbols: dict[str, int]) -> None:
    shape = _tensor_shape(value)
    kind = spec.get("kind", "unknown")
    if kind == "unknown":
        return
    if kind == "scalar":
        if len(shape) != 0:
            raise InvalidShapeError(f"{name} must be scalar, got shape {shape}")
        return
    if kind == "fixed":
        expected = tuple(spec.get("dims", []))
        if shape != expected:
            raise InvalidShapeError(f"{name} must have shape {expected}, got {shape}")
        return
    if kind == "same_as":
        other_name = spec.get("argument")
        other = tensors.get(other_name)
        if other is not None and shape != _tensor_shape(other):
            raise InvalidShapeError(f"{name} must have the same shape as {other_name}")
        return
    if kind == "parametric":
        dims = spec.get("dims", [])
        if len(shape) != len(dims):
            raise InvalidShapeError(f"{name} rank mismatch: expected {len(dims)}, got {len(shape)}")
        for symbol, actual in zip(dims, shape):
            if symbol in symbols and symbols[symbol] != actual:
                raise InvalidShapeError(f"dimension symbol {symbol} mismatch for {name}")
            symbols[symbol] = actual

def _check_backend_support(binding: dict[str, Any], tensors: dict[str, Any], scalars: dict[str, Any]) -> None:
    backend = _requested_backend(tensors, scalars)
    dtypes = [d for d in (_tensor_dtype(t) for t in tensors.values()) if d is not None]
    supports_rows = binding.get("backendSupport", [])
    if backend is not None:
        backend_rows = [s for s in supports_rows if s.get("backend") == backend]
        if not backend_rows:
            raise UnsupportedBackendError(f"{binding.get('module')}.{binding.get('pythonName')} does not support backend {backend}")
    else:
        backend_rows = supports_rows
    for dtype in dtypes:
        if not any(_dtype_supported(dtype, row.get("dtypes", [])) for row in backend_rows):
            raise InvalidDTypeError(f"{binding.get('module')}.{binding.get('pythonName')} does not support dtype {dtype}")

def _value(name: str, tensors: dict[str, Any], scalars: dict[str, Any]):
    if name in scalars:
        return scalars[name]
    if name in tensors:
        return tensors[name]
    return None

def _compare(lhs, op: str, rhs) -> bool:
    if op == "eq":
        return lhs == rhs
    if op == "ne":
        return lhs != rhs
    if op == "lt":
        return lhs < rhs
    if op == "le":
        return lhs <= rhs
    if op == "gt":
        return lhs > rhs
    if op == "ge":
        return lhs >= rhs
    raise InvalidArgumentError(f"Unknown comparison op: {op}")

def _eval_precondition(expr: dict[str, Any], tensors: dict[str, Any], scalars: dict[str, Any]) -> bool:
    kind = expr.get("kind")
    if kind == "bool_literal":
        return bool(expr.get("value"))
    if kind == "shape_equals":
        return _tensor_shape(tensors.get(expr.get("lhs"))) == _tensor_shape(tensors.get(expr.get("rhs")))
    if kind == "dtype_equals":
        return _tensor_dtype(tensors.get(expr.get("lhs"))) == _tensor_dtype(tensors.get(expr.get("rhs")))
    if kind == "backend_in":
        backend = scalars.get(expr.get("argument"))
        return str(backend) in set(expr.get("allowed", []))
    if kind == "scalar_compare":
        return _compare(_value(expr.get("lhs"), tensors, scalars), expr.get("op"), _value(expr.get("rhs"), tensors, scalars))
    if kind == "and":
        return _eval_precondition(expr.get("lhs", {}), tensors, scalars) and _eval_precondition(expr.get("rhs", {}), tensors, scalars)
    if kind == "or":
        return _eval_precondition(expr.get("lhs", {}), tensors, scalars) or _eval_precondition(expr.get("rhs", {}), tensors, scalars)
    if kind == "not":
        return not _eval_precondition(expr.get("expr", {}), tensors, scalars)
    raise InvalidArgumentError(f"Unknown precondition expression kind: {kind}")

def validate_call(module: str, name: str, tensors: dict[str, Any], scalars: dict[str, Any]) -> None:
    binding = _binding_for(module, name)
    if binding is None:
        raise RuntimeFailureError(f"No Python binding metadata for {module}.{name}")
    symbols: dict[str, int] = {}
    for tensor_spec in binding.get("tensors", []):
        tname = tensor_spec.get("name")
        value = tensors.get(tname)
        if value is None:
            if tensor_spec.get("optional", False):
                continue
            raise InvalidArgumentError(f"Missing required tensor argument {tname}")
        _check_dtype_constraint(tname, value, tensor_spec.get("dtype", {"kind": "any"}))
        _check_shape_constraint(tname, value, tensor_spec.get("shape", {"kind": "unknown"}), tensors, symbols)
        if tensor_spec.get("contiguity") in {"c_contiguous", "f_contiguous"} and getattr(value, "is_contiguous", True) is False:
            raise NonContiguousError(f"{tname} must be contiguous")
        if getattr(value, "readonly", False) and tensor_spec.get("role") in {"inout", "output"}:
            raise PermissionError(f"{tname} is read-only but binding {module}.{name} requires write access")
    _check_backend_support(binding, tensors, scalars)
    for precondition in binding.get("preconditions", []):
        if not _eval_precondition(precondition.get("expr", {}), tensors, scalars):
            raise _exception_for_code(precondition.get("errorCode"))(precondition.get("message", "precondition failed"))

@dataclass(frozen=True)
class Device:
    backend: str = "auto"
    ordinal: int = 0

    def create_queue(self, *, priority: int = -1) -> "Queue":
        return Queue(device=self, priority=priority)

    def create_event(self) -> "Event":
        return Event(device=self)

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

def _normalize_device(device: Any | None) -> Device:
    if device is None:
        return Device()
    if isinstance(device, Device):
        return device
    if isinstance(device, str):
        return Device(device)
    backend = getattr(device, "backend", None)
    if backend is not None:
        return Device(str(backend), int(getattr(device, "ordinal", 0)))
    raise InvalidArgumentError(f"Expected Device-compatible object, got {type(device).__name__}")

@dataclass(frozen=True)
class DLPackAdapter:
    """Backend-neutral DLPack adapter facade for zero-copy import/export checks."""
    device: Device = field(default_factory=Device)

    def supports_zero_copy_import(self, producer, *, dtype: str | None = None) -> bool:
        return can_zero_copy_dlpack(producer, self.device, dtype=dtype)

    def supports_zero_copy_export(self, consumer=None, *, dtype: str | None = None) -> bool:
        target = self.device if consumer is None else consumer
        return can_zero_copy_dlpack(self.device, target, dtype=dtype)

    def from_dlpack(self, obj, *, queue: "Queue | None" = None) -> "Tensor":
        return Tensor.from_dlpack(obj, device=self.device, queue=queue)

    def to_dlpack(self, tensor: "Tensor", *, stream=None, max_version=None):
        if not self.supports_zero_copy_export(tensor.device, dtype=tensor.dtype):
            raise RuntimeError(f"Cannot zero-copy DLPack tensor from {tensor.device.backend} through adapter for {self.device.backend}; use explicit copy")
        return tensor.__dlpack__(stream=stream, max_version=max_version)

@dataclass
class Queue:
    device: Device = field(default_factory=Device)
    priority: int = -1
    _handle: Optional[Any] = None
    _closed: bool = False

    def _ensure_handle(self):
        if self._closed:
            raise RuntimeFailureError("Queue is closed")
        if self._handle is None:
            ext = _core._load_extension(required=True)
            if not hasattr(ext, "runtime") or not hasattr(ext.runtime, "queue_create"):
                raise RuntimeFailureError("Queue creation is not linked in this navatala_gpu build")
            self._handle = ext.runtime.queue_create(self.device, int(self.priority))
        return self._handle

    def sync(self) -> None:
        if self._closed or self._handle is None:
            return
        ext = _core._load_extension(required=True)
        if hasattr(ext, "runtime") and hasattr(ext.runtime, "queue_sync"):
            ext.runtime.queue_sync(self)

    def close(self) -> None:
        if self._closed:
            return
        ext = _core._load_extension(required=False)
        if self._handle is not None and ext is not None and hasattr(ext, "runtime") and hasattr(ext.runtime, "queue_close"):
            ext.runtime.queue_close(self._handle)
        self._handle = None
        self._closed = True

    def __enter__(self):
        self._ensure_handle()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            try:
                _LOG.debug("Queue finalizer cleanup failed", exc_info=True)
            except Exception:
                pass

@dataclass
class Event:
    device: Device = field(default_factory=Device)
    _handle: Optional[Any] = None
    _closed: bool = False

    def _ensure_handle(self, queue: Queue | None = None):
        if self._closed:
            raise RuntimeFailureError("Event is closed")
        if self._handle is None:
            ext = _core._load_extension(required=True)
            if not hasattr(ext, "runtime") or not hasattr(ext.runtime, "event_create"):
                raise RuntimeFailureError("Event creation is not linked in this navatala_gpu build")
            if queue is not None and hasattr(ext.runtime, "event_create_from_queue"):
                self._handle = ext.runtime.event_create_from_queue(queue)
            else:
                self._handle = ext.runtime.event_create(self.device)
        return self._handle

    def record(self, queue: Queue) -> None:
        self._ensure_handle(queue)
        ext = _core._load_extension(required=True)
        ext.runtime.event_record(self, queue)

    def wait(self, queue: Queue) -> None:
        self._ensure_handle(queue)
        ext = _core._load_extension(required=True)
        ext.runtime.event_wait(queue, self)

    def sync(self) -> None:
        if self._closed or self._handle is None:
            return
        ext = _core._load_extension(required=True)
        ext.runtime.event_sync(self)

    synchronize = sync

    def is_complete(self) -> bool:
        if self._closed or self._handle is None:
            return True
        ext = _core._load_extension(required=True)
        return bool(ext.runtime.event_is_complete(self))

    def close(self) -> None:
        if self._closed:
            return
        ext = _core._load_extension(required=False)
        if self._handle is not None and ext is not None and hasattr(ext, "runtime") and hasattr(ext.runtime, "event_close"):
            ext.runtime.event_close(self._handle)
        self._handle = None
        self._closed = True

    def __enter__(self):
        self._ensure_handle()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            try:
                _LOG.debug("Event finalizer cleanup failed", exc_info=True)
            except Exception:
                pass

class Buffer:
    def __init__(self, handle: Any = None, *, nbytes: int = 0, device: Device | None = None):
        self._handle = handle
        self._nbytes = int(nbytes)
        self.device = device or Device()
        self._closed = False

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def closed(self) -> bool:
        return self._closed

    @classmethod
    def empty(cls, nbytes: int, *, device: Device | None = None, queue: Queue | None = None):
        if int(nbytes) < 0:
            raise InvalidArgumentError("Buffer.empty requires a non-negative byte count")
        ext = _core._load_extension(required=True)
        if not hasattr(ext, "runtime") or not hasattr(ext.runtime, "buffer_empty"):
            raise NotImplementedError("Buffer.empty is not linked in this navatala_gpu build")
        return ext.runtime.buffer_empty(int(nbytes), _normalize_device(device), queue)

    def close(self) -> None:
        if self._closed:
            return
        ext = _core._load_extension(required=False)
        if self._handle is not None and ext is not None and hasattr(ext, "runtime") and hasattr(ext.runtime, "buffer_close"):
            ext.runtime.buffer_close(self._handle)
        self._handle = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            try:
                _LOG.debug("Buffer finalizer cleanup failed", exc_info=True)
            except Exception:
                pass

class Tensor:
    def __init__(self, handle: Any = None, *, shape=(), dtype="float32", device: Device | None = None, readonly: bool = False):
        self._handle = handle
        self.shape = tuple(shape)
        self.dtype = str(dtype)
        self.device = device or Device()
        self.readonly = bool(readonly)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @classmethod
    def empty(cls, shape, dtype="float32", device: Device | None = None, queue: Queue | None = None):
        ext = _core._load_extension(required=True)
        if not hasattr(ext, "runtime") or not hasattr(ext.runtime, "tensor_empty"):
            raise NotImplementedError("Tensor.empty is not linked in this navatala_gpu build")
        return ext.runtime.tensor_empty(tuple(shape), str(dtype), _normalize_device(device), queue)

    @classmethod
    def empty_like(cls, other: "Tensor", *, queue: Queue | None = None):
        return cls.empty(other.shape, dtype=other.dtype, device=other.device, queue=queue)

    @classmethod
    def from_numpy(cls, array, *, device: Device | None = None, queue: Queue | None = None):
        import numpy as np
        arr = np.asarray(array)
        if arr.dtype.kind in {"c", "O", "b"} or str(arr.dtype) == "float128":
            raise TypeError(f"navatala_gpu does not support NumPy dtype {arr.dtype}")
        ext = _core._load_extension(required=True)
        if not hasattr(ext, "runtime") or not hasattr(ext.runtime, "tensor_from_numpy"):
            raise NotImplementedError("Tensor.from_numpy is not linked in this navatala_gpu build")
        return ext.runtime.tensor_from_numpy(arr, _normalize_device(device), queue)

    @classmethod
    def from_dlpack(cls, obj, *, device: Device | None = None, queue: Queue | None = None):
        ext = _core._load_extension(required=True)
        if not hasattr(ext, "runtime") or not hasattr(ext.runtime, "tensor_from_dlpack"):
            raise NotImplementedError("Tensor.from_dlpack is not linked in this navatala_gpu build")
        return ext.runtime.tensor_from_dlpack(obj, device, queue)

    def to_numpy(self, *, queue: Queue | None = None):
        ext = _core._load_extension(required=True)
        if not hasattr(ext, "runtime") or not hasattr(ext.runtime, "tensor_to_numpy"):
            raise NotImplementedError("Tensor.to_numpy is not linked in this navatala_gpu build")
        return ext.runtime.tensor_to_numpy(self, queue)

    def __dlpack__(self, stream=None, max_version=None):
        ext = _core._load_extension(required=True)
        if not hasattr(ext, "runtime") or not hasattr(ext.runtime, "tensor_to_dlpack"):
            raise NotImplementedError("DLPack export is not linked in this navatala_gpu build")
        return ext.runtime.tensor_to_dlpack(self, stream, max_version)

    def to_dlpack(self, stream=None, max_version=None):
        return self.__dlpack__(stream=stream, max_version=max_version)

    def __dlpack_device__(self):
        ext = _core._load_extension(required=True)
        if not hasattr(ext, "runtime") or not hasattr(ext.runtime, "tensor_dlpack_device"):
            raise NotImplementedError("DLPack device query is not linked in this navatala_gpu build")
        return ext.runtime.tensor_dlpack_device(self)

    def close(self) -> None:
        if self._closed:
            return
        ext = _core._load_extension(required=False)
        if ext is not None and hasattr(ext, "runtime") and hasattr(ext.runtime, "tensor_close"):
            ext.runtime.tensor_close(self)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            try:
                _LOG.debug("Tensor finalizer cleanup failed", exc_info=True)
            except Exception:
                pass

def get_capabilities() -> dict[str, Any]:
    ext = _core._load_extension(required=False)
    if ext is not None and hasattr(ext, "get_capabilities"):
        caps = ext.get_capabilities()
    else:
        caps = {
        "extension_loaded": False,
        "extension_error": repr(_core.extension_error()),
        "backends": {backend: {"compiled": True, "available": False, "initialized": False, "selected": False, "memory": {"supported": False, "free_bytes": 0, "total_bytes": 0, "error_code": "not_loaded"}} for backend in _KNOWN_BACKENDS},
        "live_dlpack_exports": 0,
        }
    caps["manifest_id"] = "pyabi6-2675513039"
    caps["abi_version"] = 6
    caps["operations"] = _operation_support()
    return caps

def supports(operation: str, *, backend: str | None = None, dtype: str | None = None) -> bool:
    for binding in _API_MANIFEST.get("bindings", []):
        public_name = f"{binding.get('module')}.{binding.get('pythonName')}"
        if operation not in {binding.get("pythonName"), public_name}:
            continue
        if backend is None and dtype is None:
            return True
        for support in binding.get("backendSupport", []):
            if backend is not None and support.get("backend") != backend:
                continue
            if dtype is not None and not _dtype_supported(dtype, support.get("dtypes", [])):
                continue
            return True
    return False
