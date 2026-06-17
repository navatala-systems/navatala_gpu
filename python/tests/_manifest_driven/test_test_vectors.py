# SPDX-License-Identifier: Apache-2.0
import json
import pathlib

import pytest

linalg = pytest.importorskip("navatala_gpu.linalg")
runtime = pytest.importorskip("navatala_gpu.runtime")

def _available_device():
    caps = runtime.get_capabilities()
    if not caps.get("extension_loaded", False):
        pytest.skip("navatala_gpu extension is not available")
    for backend, status in caps.get("backends", {}).items():
        if status.get("available", False):
            return runtime.Device(backend)
    pytest.skip("no navatala_gpu backend is available")

def _vectors():
    vector_path = pathlib.Path(__file__).resolve().parents[1] / "vectors" / "basic_linalg.json"
    return json.loads(vector_path.read_text(encoding="utf-8"))["vectors"]

def test_linalg_vector_corpus_executes_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    vectors = _vectors()
    with runtime.Queue(device=device) as queue:
        if hasattr(linalg, "axpy"):
            case = vectors["axpy_f32"]
            x = runtime.Tensor.from_numpy(np.asarray(case["x"], dtype=np.float32), device=device, queue=queue)
            y = runtime.Tensor.from_numpy(np.asarray(case["y"], dtype=np.float32), device=device, queue=queue)
            out = linalg.axpy(x, y, alpha=float(case["alpha"]), backend=device.backend, queue=queue, sync=True)
            np.testing.assert_allclose(out.to_numpy(queue=queue), np.asarray(case["expected"], dtype=np.float32), rtol=1e-6)
        if hasattr(linalg, "gemm"):
            case = vectors["gemm_f32"]
            a = runtime.Tensor.from_numpy(np.asarray(case["a"], dtype=np.float32), device=device, queue=queue)
            b = runtime.Tensor.from_numpy(np.asarray(case["b"], dtype=np.float32), device=device, queue=queue)
            c = runtime.Tensor.from_numpy(np.asarray(case["c"], dtype=np.float32), device=device, queue=queue)
            out = linalg.gemm(a, b, c, alpha=float(case["alpha"]), beta=float(case["beta"]), backend=device.backend, queue=queue, sync=True)
            np.testing.assert_allclose(out.to_numpy(queue=queue), np.asarray(case["expected"], dtype=np.float32), rtol=1e-6)
        if hasattr(linalg, "nrm2"):
            case = vectors["nrm2_f32"]
            x = runtime.Tensor.from_numpy(np.asarray(case["x"], dtype=np.float32), device=device, queue=queue)
            result = runtime.Tensor.empty((), dtype="float32", device=device, queue=queue)
            out = linalg.nrm2(x, result, backend=device.backend, queue=queue, sync=True)
            np.testing.assert_allclose(np.asarray(out.to_numpy(queue=queue)).item(), np.float32(case["expected"]), rtol=1e-6)
