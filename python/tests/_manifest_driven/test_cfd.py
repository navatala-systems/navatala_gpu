# SPDX-License-Identifier: Apache-2.0
import importlib

def test_cfd_exports_are_importable():
    mod = importlib.import_module("navatala_gpu.cfd")
    assert hasattr(mod, "__all__")
    assert hasattr(mod, "list_bindings")
    assert hasattr(mod, "supports")
    listed = mod.list_bindings()
    assert isinstance(listed, tuple)
    assert listed == ()
    for public_name in listed:
        assert public_name in mod.__all__
        assert mod.supports(public_name) is True
    assert mod.supports("__missing_operation__") is False
