# SPDX-License-Identifier: Apache-2.0
from ._core import call_extension
from .runtime import supports as _runtime_supports, validate_call

__navatala_module__ = "cfd"
_BINDINGS = ()
__all__ = ("list_bindings", "supports")

def list_bindings() -> tuple[str, ...]:
    """Return the public operation names exposed by this module."""
    return _BINDINGS

def supports(operation: str, *, backend: str | None = None, dtype: str | None = None) -> bool:
    """Return whether this module exposes an operation for the requested backend/dtype."""
    if operation not in _BINDINGS:
        return False
    return _runtime_supports(f"{__navatala_module__}.{operation}", backend=backend, dtype=dtype)
