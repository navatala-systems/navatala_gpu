# SPDX-License-Identifier: Apache-2.0
from ._core import call_extension
from .runtime import InvalidArgumentError, Queue

__all__ = ["launch"]

def _triple(value, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        value = (value,)
    dims = tuple(int(v) for v in value)
    if not 1 <= len(dims) <= 3:
        raise InvalidArgumentError(f"{name} must have one to three dimensions")
    if any(v <= 0 for v in dims):
        raise InvalidArgumentError(f"{name} dimensions must be positive")
    return dims + (1,) * (3 - len(dims))

def launch(source, entry_point: str, args=(), *, source_kind: str = "opencl_c", queue: Queue | None = None, grid=(1,), block=(1,), sync: bool = False):
    """Internal/advanced raw kernel launch helper.

    Stable public APIs should prefer the operation wrappers in the dotted
    submodules. This helper is intended for kernel-level smoke tests
    requires an explicit Queue so context ownership is unambiguous.
    """
    if queue is None:
        raise InvalidArgumentError("kernels.launch requires an explicit Queue")
    if not entry_point:
        raise InvalidArgumentError("kernels.launch requires a non-empty entry_point")
    return call_extension(
        "kernels",
        "launch",
        source,
        str(entry_point),
        list(args),
        source_kind=str(source_kind),
        queue=queue,
        grid=list(_triple(grid, "grid")),
        block=list(_triple(block, "block")),
        sync=bool(sync),
    )
