# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
from ._exceptions import MissingExtensionError

_EXPECTED_ABI_VERSION = 6
_EXTENSION_MODULE = "navatala_gpu_ext"
_ext = None
_extension_error = None

def _load_extension(required: bool = False):
    global _ext, _extension_error
    if _ext is not None:
        return _ext
    try:
        mod = importlib.import_module(f"{__package__}.{_EXTENSION_MODULE}")
    except Exception as package_exc:
        try:
            mod = importlib.import_module(_EXTENSION_MODULE)
        except Exception as fallback_exc:
            _extension_error = package_exc
            if required:
                raise MissingExtensionError(
                    f"navatala_gpu extension '{_EXTENSION_MODULE}' is not available: {package_exc}"
                ) from package_exc
            return None
    abi = getattr(mod, "__navatala_ffi_abi_version__", None)
    if abi is None and hasattr(mod, "_get_abi_version"):
        abi = mod._get_abi_version()
    if abi is not None and int(abi) != _EXPECTED_ABI_VERSION:
        raise ImportError(
            f"navatala_gpu ABI mismatch: Python expects {_EXPECTED_ABI_VERSION}, extension reports {abi}"
        )
    _ext = mod
    return mod

def has_extension() -> bool:
    return _load_extension(required=False) is not None

def extension_error():
    _load_extension(required=False)
    return _extension_error

def call_extension(module: str, name: str, *args, **kwargs):
    mod = _load_extension(required=True)
    submodule = getattr(mod, module, None)
    if submodule is None:
        raise NotImplementedError(f"navatala_gpu extension has no module '{module}'")
    fn = getattr(submodule, name, None)
    if fn is None:
        raise NotImplementedError(f"navatala_gpu operation '{module}.{name}' is not available")
    return fn(*args, **kwargs)

_load_extension(required=False)
