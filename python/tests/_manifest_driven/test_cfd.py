# SPDX-License-Identifier: Apache-2.0
import importlib

def test_cfd_exports_are_importable():
    pkg = importlib.import_module("navatala_gpu")
    mod = importlib.import_module("navatala_gpu.cfd")
    assert hasattr(mod, "__all__")
    assert hasattr(mod, "list_bindings")
    assert hasattr(mod, "supports")
    listed = mod.list_bindings()
    assert isinstance(listed, tuple)
    assert listed == ()
    for public_name in listed:
        assert public_name in mod.__all__
        caps = pkg.get_capabilities()
        key = "cfd" + "." + public_name
        expected = bool(caps.get("operations", {}).get(key, {}).get("backends", {}))
        assert mod.supports(public_name) is expected
    assert mod.supports("__missing_operation__") is False
