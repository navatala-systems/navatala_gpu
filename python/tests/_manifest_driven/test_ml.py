# SPDX-License-Identifier: Apache-2.0
import importlib

def test_ml_exports_are_importable():
    mod = importlib.import_module("navatala_gpu.ml")
    assert hasattr(mod, "mean_squared_error")
    assert hasattr(mod, "root_mean_squared_error")
    assert hasattr(mod, "mean_absolute_error")
    assert hasattr(mod, "mean_absolute_percentage_error")
    assert hasattr(mod, "r2_score")
    assert hasattr(mod, "explained_variance_score")
    assert hasattr(mod, "__all__")
    assert hasattr(mod, "list_bindings")
    assert hasattr(mod, "supports")
    listed = mod.list_bindings()
    assert isinstance(listed, tuple)
    assert listed == ("mean_squared_error", "root_mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "r2_score", "explained_variance_score")
    for public_name in listed:
        assert public_name in mod.__all__
        assert mod.supports(public_name) is True
    assert mod.supports("__missing_operation__") is False

def test_ml_unsupported_backend_fails_before_extension_load():
    mod = importlib.import_module("navatala_gpu.ml")
    from navatala_gpu import runtime
    try:
        mod.mean_squared_error(runtime.Tensor(shape=(1,), dtype="Float32", device=runtime.Device("unsupported")), runtime.Tensor(shape=(1,), dtype="Float32", device=runtime.Device("unsupported")), runtime.Tensor(shape=(), dtype="Float32", device=runtime.Device("unsupported")), 1, "auto")
    except runtime.UnsupportedBackendError:
        pass
    else:
        raise AssertionError("expected UnsupportedBackendError")
