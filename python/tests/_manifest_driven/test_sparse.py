# SPDX-License-Identifier: Apache-2.0
import importlib

def test_sparse_exports_are_importable():
    mod = importlib.import_module("navatala_gpu.sparse")
    assert hasattr(mod, "csr_spmv")
    assert hasattr(mod, "__all__")
    assert hasattr(mod, "list_bindings")
    assert hasattr(mod, "supports")
    listed = mod.list_bindings()
    assert isinstance(listed, tuple)
    assert listed == ("csr_spmv",)
    for public_name in listed:
        assert public_name in mod.__all__
        assert mod.supports(public_name) is True
    assert mod.supports("__missing_operation__") is False

def test_sparse_unsupported_backend_fails_before_extension_load():
    mod = importlib.import_module("navatala_gpu.sparse")
    from navatala_gpu import runtime
    try:
        mod.csr_spmv(runtime.Tensor(shape=(1,), dtype="UInt32", device=runtime.Device("unsupported")), runtime.Tensor(shape=(1,), dtype="UInt32", device=runtime.Device("unsupported")), runtime.Tensor(shape=(1,), dtype="Float32", device=runtime.Device("unsupported")), runtime.Tensor(shape=(1,), dtype="Float32", device=runtime.Device("unsupported")), runtime.Tensor(shape=(1,), dtype="Float32", device=runtime.Device("unsupported")), 1, 1, 1, "auto")
    except runtime.UnsupportedBackendError:
        pass
    else:
        raise AssertionError("expected UnsupportedBackendError")
