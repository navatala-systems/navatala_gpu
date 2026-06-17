# SPDX-License-Identifier: Apache-2.0
import importlib

def test_dataframe_exports_are_importable():
    mod = importlib.import_module("navatala_gpu.dataframe")
    assert hasattr(mod, "reduce_sum")
    assert hasattr(mod, "reduce_sum_i32_to_i64")
    assert hasattr(mod, "reduce_min_i32")
    assert hasattr(mod, "reduce_max_i32")
    assert hasattr(mod, "fill_null")
    assert hasattr(mod, "fill_null_i32")
    assert hasattr(mod, "gather")
    assert hasattr(mod, "gather_i32")
    assert hasattr(mod, "scatter")
    assert hasattr(mod, "scatter_i32")
    assert hasattr(mod, "compact_by_mask")
    assert hasattr(mod, "compact_by_mask_i32")
    assert hasattr(mod, "coalesce")
    assert hasattr(mod, "copy_if_valid")
    assert hasattr(mod, "inclusive_scan_sum_valid_prefix")
    assert hasattr(mod, "inclusive_scan_sum_i32_to_i64_skip_nulls")
    assert hasattr(mod, "count_valid")
    assert hasattr(mod, "is_valid")
    assert hasattr(mod, "is_null")
    assert hasattr(mod, "fill_constant")
    assert hasattr(mod, "fill_constant_i32")
    assert hasattr(mod, "fill_range")
    assert hasattr(mod, "fill_validity_all_valid")
    assert hasattr(mod, "fill_validity_all_null")
    assert hasattr(mod, "__all__")
    assert hasattr(mod, "list_bindings")
    assert hasattr(mod, "supports")
    listed = mod.list_bindings()
    assert isinstance(listed, tuple)
    assert listed == ("reduce_sum", "reduce_sum_i32_to_i64", "reduce_min_i32", "reduce_max_i32", "fill_null", "fill_null_i32", "gather", "gather_i32", "scatter", "scatter_i32", "compact_by_mask", "compact_by_mask_i32", "coalesce", "copy_if_valid", "inclusive_scan_sum_valid_prefix", "inclusive_scan_sum_i32_to_i64_skip_nulls", "count_valid", "is_valid", "is_null", "fill_constant", "fill_constant_i32", "fill_range", "fill_validity_all_valid", "fill_validity_all_null")
    for public_name in listed:
        assert public_name in mod.__all__
        assert mod.supports(public_name) is True
    assert mod.supports("__missing_operation__") is False

def test_dataframe_unsupported_backend_fails_before_extension_load():
    mod = importlib.import_module("navatala_gpu.dataframe")
    from navatala_gpu import runtime
    try:
        mod.reduce_sum(runtime.Tensor(shape=(1,), dtype="Float32", device=runtime.Device("unsupported")), runtime.Tensor(shape=(1,), dtype="UInt32", device=runtime.Device("unsupported")), runtime.Tensor(shape=(), dtype="Float32", device=runtime.Device("unsupported")), runtime.Tensor(shape=(1,), dtype="UInt32", device=runtime.Device("unsupported")), 1, "auto")
    except runtime.UnsupportedBackendError:
        pass
    else:
        raise AssertionError("expected UnsupportedBackendError")
