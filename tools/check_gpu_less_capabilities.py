#!/usr/bin/env python3
import navatala_gpu

caps = navatala_gpu.get_capabilities()
assert caps.get("extension_loaded") is True, "native extension did not load"
assert caps.get("runtime_mode") in ("real", "stub"), "unknown runtime mode"
backends = caps.get("backends")
assert isinstance(backends, dict) and set(backends) == {"cuda", "hip", "metal", "opencl", "vulkan"}, "incomplete backend inventory"
for name, status in backends.items():
    assert status.get("available") is False, (name, "fabricated availability")
    assert status.get("initialized") is False, (name, "fabricated initialization")
    assert status.get("selected") is False, (name, "fabricated selection")
    assert status.get("device_name") == "", (name, "device name on GPU-less runner")
    memory = status.get("memory", {})
    assert memory.get("supported") is False, (name, "fabricated memory support")
    assert memory.get("free_bytes") == 0 and memory.get("total_bytes") == 0, (name, "fabricated memory capacity")
    if caps["runtime_mode"] == "stub":
        assert status.get("compiled") is False, (name, "stub reports vendor backend")
operations = caps.get("operations")
assert isinstance(operations, dict) and operations, "missing operation inventory"
for name in operations:
    assert not navatala_gpu.supports(name), (name, "execution advertised without device")
print("GPU-less native capability truthfulness passed:", caps["runtime_mode"])
