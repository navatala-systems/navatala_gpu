# SPDX-License-Identifier: Apache-2.0
import pytest

linalg = pytest.importorskip("navatala_gpu.linalg")
dataframe = pytest.importorskip("navatala_gpu.dataframe")
sparse = pytest.importorskip("navatala_gpu.sparse")
graph = pytest.importorskip("navatala_gpu.graph")
ml = pytest.importorskip("navatala_gpu.ml")
kernels = pytest.importorskip("navatala_gpu.kernels")
runtime = pytest.importorskip("navatala_gpu.runtime")

def _available_device():
    caps = runtime.get_capabilities()
    if not caps.get("extension_loaded", False):
        pytest.skip("navatala_gpu extension is not available")
    for backend, status in caps.get("backends", {}).items():
        if status.get("available", False):
            return runtime.Device(backend)
    pytest.skip("no navatala_gpu backend is available")

def test_programmatic_memory_telemetry_shape():
    caps = runtime.get_capabilities()
    assert "backends" in caps
    for backend, status in caps.get("backends", {}).items():
        memory = status.get("memory")
        assert isinstance(memory, dict), backend
        assert {"supported", "free_bytes", "total_bytes", "error_code"}.issubset(memory), backend
        assert isinstance(memory["supported"], bool)
        assert memory["free_bytes"] >= 0
        assert memory["total_bytes"] >= 0
        if memory["supported"]:
            assert memory["total_bytes"] >= memory["free_bytes"]

def test_numpy_roundtrip_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    src = np.arange(8, dtype=np.float32)
    tensor = runtime.Tensor.from_numpy(src, device=device)
    out = tensor.to_numpy()
    np.testing.assert_array_equal(out, src)

def test_numpy_scalar_roundtrip_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    src = np.asarray(3.5, dtype=np.float32)
    tensor = runtime.Tensor.from_numpy(src, device=device)
    assert tuple(tensor.shape) == ()
    out = tensor.to_numpy()
    assert out.shape == ()
    np.testing.assert_array_equal(out, src)

def test_queue_backed_copy_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    with runtime.Queue(device=device) as queue:
        src = np.arange(6, dtype=np.float32)
        tensor = runtime.Tensor.from_numpy(src, device=device, queue=queue)
        queue.sync()
        np.testing.assert_array_equal(tensor.to_numpy(queue=queue), src)

def test_event_record_wait_if_backend_available():
    device = _available_device()
    with runtime.Queue(device=device) as queue:
        event = runtime.Event(device=device)
        event.record(queue)
        event.wait(queue)
        event.sync()
        assert isinstance(event.is_complete(), bool)
        event.close()

def test_device_queue_and_buffer_if_backend_available():
    device = _available_device()
    with device:
        with device.create_queue() as queue:
            queue.sync()
        buffer = runtime.Buffer.empty(1024, device=device)
        assert buffer.nbytes == 1024
        buffer.close()
        assert getattr(buffer, "closed", getattr(buffer, "_closed", False))

def test_axpy_dispatch_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    x = runtime.Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32), device=device)
    y = runtime.Tensor.from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32), device=device)
    with runtime.Queue(device=device) as queue:
        out = linalg.axpy(x, y, alpha=2.5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), np.array([12.5, 25.0, 37.5], dtype=np.float32), rtol=1e-6)
    with pytest.raises(runtime.InvalidArgumentError, match="sync=True requires an explicit Queue"):
        linalg.axpy(x, y, alpha=1.0, backend=device.backend, sync=True)

def test_axpy_dispatch_group_if_backend_available():
    if not hasattr(linalg, "axpy_dispatch"):
        pytest.skip("no dispatch-group AXPY binding in this build")
    np = pytest.importorskip("numpy")
    device = _available_device()
    x = runtime.Tensor.from_numpy(np.array([2.0, 4.0, 6.0], dtype=np.float32), device=device)
    y = runtime.Tensor.from_numpy(np.array([1.0, 1.0, 1.0], dtype=np.float32), device=device)
    with runtime.Queue(device=device) as queue:
        out = linalg.axpy_dispatch(x, y, alpha=0.5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), np.array([2.0, 3.0, 4.0], dtype=np.float32), rtol=1e-6)

def test_gemm_dispatch_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b_np = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
    c_np = np.ones((2, 2), dtype=np.float32)
    a = runtime.Tensor.from_numpy(a_np, device=device)
    b = runtime.Tensor.from_numpy(b_np, device=device)
    c = runtime.Tensor.from_numpy(c_np, device=device)
    with runtime.Queue(device=device) as queue:
        out = linalg.gemm(a, b, c, alpha=1.0, beta=0.5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), a_np @ b_np + 0.5 * c_np, rtol=1e-6)

def test_nrm2_dispatch_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(linalg, "nrm2"):
        pytest.skip("no NRM2 binding in this build")
    x_np = np.array([3.0, 4.0, 12.0], dtype=np.float32)
    x = runtime.Tensor.from_numpy(x_np, device=device)
    result = runtime.Tensor.empty((), dtype="float32", device=device)
    with runtime.Queue(device=device) as queue:
        out = linalg.nrm2(x, result, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), np.linalg.norm(x_np).astype(np.float32), rtol=1e-6)

def test_dataframe_reduce_sum_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "reduce_sum"):
        pytest.skip("no DataFrame reduce_sum binding in this build")
    data_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_valid_np = np.array([0b1011], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        output = runtime.Tensor.empty((), dtype="float32", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.reduce_sum(data, input_valid, output, output_valid, count=4, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), np.array(7.0, dtype=np.float32), rtol=1e-6)
    np.testing.assert_array_equal(output_valid.to_numpy(), np.array([1], dtype=np.uint32))

def test_dataframe_reduce_sum_i32_to_i64_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "reduce_sum_i32_to_i64"):
        pytest.skip("no DataFrame reduce_sum_i32_to_i64 binding in this build")
    data_np = np.array([2147483647, 1, 100, 10], dtype=np.int32)
    input_valid_np = np.array([0b1011], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        output = runtime.Tensor.empty((), dtype="int64", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.reduce_sum_i32_to_i64(data, input_valid, output, output_valid, count=4, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), np.array(2147483658, dtype=np.int64))
    np.testing.assert_array_equal(output_valid.to_numpy(), np.array([1], dtype=np.uint32))

def test_dataframe_reduce_min_i32_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "reduce_min_i32"):
        pytest.skip("no DataFrame reduce_min_i32 binding in this build")
    data_np = np.array([5, -2, 99, 12, -9], dtype=np.int32)
    input_valid_np = np.array([0b01011], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        output = runtime.Tensor.empty((), dtype="int32", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.reduce_min_i32(data, input_valid, output, output_valid, count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), np.array(-2, dtype=np.int32))
    np.testing.assert_array_equal(output_valid.to_numpy(), np.array([1], dtype=np.uint32))

def test_dataframe_reduce_max_i32_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "reduce_max_i32"):
        pytest.skip("no DataFrame reduce_max_i32 binding in this build")
    data_np = np.array([5, -2, 99, 12, -9], dtype=np.int32)
    input_valid_np = np.array([0b01011], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        output = runtime.Tensor.empty((), dtype="int32", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.reduce_max_i32(data, input_valid, output, output_valid, count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), np.array(12, dtype=np.int32))
    np.testing.assert_array_equal(output_valid.to_numpy(), np.array([1], dtype=np.uint32))

def test_dataframe_fill_null_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "fill_null"):
        pytest.skip("no DataFrame fill_null binding in this build")
    data_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_valid_np = np.array([0b1011], dtype=np.uint32)
    expected = np.array([1.0, 2.0, -5.0, 4.0], dtype=np.float32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        output = runtime.Tensor.empty((4,), dtype="float32", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.fill_null(data, input_valid, output, output_valid, fill_value=-5.0, count=4, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)
    np.testing.assert_array_equal(output_valid.to_numpy(), np.array([0b1111], dtype=np.uint32))

def test_dataframe_fill_null_i32_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "fill_null_i32"):
        pytest.skip("no DataFrame fill_null_i32 binding in this build")
    data_np = np.array([10, 20, 30, 40], dtype=np.int32)
    input_valid_np = np.array([0b1011], dtype=np.uint32)
    expected = np.array([10, 20, -9, 40], dtype=np.int32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        output = runtime.Tensor.empty((4,), dtype="int32", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.fill_null_i32(data, input_valid, output, output_valid, fill_value=-9, count=4, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    np.testing.assert_array_equal(output_valid.to_numpy(), np.array([0b1111], dtype=np.uint32))

def test_dataframe_gather_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "gather"):
        pytest.skip("no DataFrame gather binding in this build")
    data_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    input_valid_np = np.array([0x0000000B], dtype=np.uint32)
    indices_np = np.array([3, 2, -1, 0, 99], dtype=np.int32)
    expected = np.array([40.0, 0.0, 0.0, 10.0, 0.0], dtype=np.float32)
    expected_valid = np.array([0x00000009], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        indices = runtime.Tensor.from_numpy(indices_np, device=device, queue=queue)
        output = runtime.Tensor.empty((5,), dtype="float32", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.gather(data, input_valid, indices, output, output_valid, input_count=4, output_count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)
    np.testing.assert_array_equal(output_valid.to_numpy(), expected_valid)

def test_dataframe_gather_i32_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "gather_i32"):
        pytest.skip("no DataFrame gather_i32 binding in this build")
    data_np = np.array([10, 20, 30, 40], dtype=np.int32)
    input_valid_np = np.array([0x0000000B], dtype=np.uint32)
    indices_np = np.array([3, 2, -1, 0, 99], dtype=np.int32)
    expected = np.array([40, 0, 0, 10, 0], dtype=np.int32)
    expected_valid = np.array([0x00000009], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        indices = runtime.Tensor.from_numpy(indices_np, device=device, queue=queue)
        output = runtime.Tensor.empty((5,), dtype="int32", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.gather_i32(data, input_valid, indices, output, output_valid, input_count=4, output_count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    np.testing.assert_array_equal(output_valid.to_numpy(), expected_valid)

def test_dataframe_scatter_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "scatter"):
        pytest.skip("no DataFrame scatter binding in this build")
    data_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    input_valid_np = np.array([0b1011], dtype=np.uint32)
    indices_np = np.array([3, 1, 2, 0], dtype=np.int32)
    output_np = np.array([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    output_valid_np = np.array([0], dtype=np.uint32)
    expected = np.array([40.0, 20.0, 30.0, 10.0, -1.0], dtype=np.float32)
    expected_valid = np.array([0b1011], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        indices = runtime.Tensor.from_numpy(indices_np, device=device, queue=queue)
        output = runtime.Tensor.from_numpy(output_np, device=device, queue=queue)
        output_valid = runtime.Tensor.from_numpy(output_valid_np, device=device, queue=queue)
        out = dataframe.scatter(data, input_valid, indices, output, output_valid, input_count=4, output_count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)
    np.testing.assert_array_equal(output_valid.to_numpy(), expected_valid)

def test_dataframe_scatter_i32_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "scatter_i32"):
        pytest.skip("no DataFrame scatter_i32 binding in this build")
    data_np = np.array([10, 20, 30, 40], dtype=np.int32)
    input_valid_np = np.array([0b1011], dtype=np.uint32)
    indices_np = np.array([3, 1, 2, 0], dtype=np.int32)
    output_np = np.array([-1, -1, -1, -1, -1], dtype=np.int32)
    output_valid_np = np.array([0], dtype=np.uint32)
    expected = np.array([40, 20, 30, 10, -1], dtype=np.int32)
    expected_valid = np.array([0b1011], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        indices = runtime.Tensor.from_numpy(indices_np, device=device, queue=queue)
        output = runtime.Tensor.from_numpy(output_np, device=device, queue=queue)
        output_valid = runtime.Tensor.from_numpy(output_valid_np, device=device, queue=queue)
        out = dataframe.scatter_i32(data, input_valid, indices, output, output_valid, input_count=4, output_count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    np.testing.assert_array_equal(output_valid.to_numpy(), expected_valid)

def test_dataframe_compact_by_mask_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "compact_by_mask"):
        pytest.skip("no DataFrame compact_by_mask binding in this build")
    data_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_valid_np = np.array([0b11011], dtype=np.uint32)
    mask_np = np.array([1, 0, 1, 1, 0], dtype=np.uint32)
    prefix_np = np.array([0, 1, 1, 2, 3], dtype=np.uint32)
    expected = np.array([1.0, 3.0, 4.0], dtype=np.float32)
    expected_valid = np.array([0b101], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        mask = runtime.Tensor.from_numpy(mask_np, device=device, queue=queue)
        prefix_sum = runtime.Tensor.from_numpy(prefix_np, device=device, queue=queue)
        output = runtime.Tensor.empty((3,), dtype="float32", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.compact_by_mask(data, input_valid, mask, prefix_sum, output, output_valid, count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)
    np.testing.assert_array_equal(output_valid.to_numpy(), expected_valid)

def test_dataframe_compact_by_mask_i32_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "compact_by_mask_i32"):
        pytest.skip("no DataFrame compact_by_mask_i32 binding in this build")
    data_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_valid_np = np.array([0b11011], dtype=np.uint32)
    mask_np = np.array([1, 0, 1, 1, 0], dtype=np.uint32)
    prefix_np = np.array([0, 1, 1, 2, 3], dtype=np.uint32)
    expected = np.array([1, 3, 4], dtype=np.int32)
    expected_valid = np.array([0b101], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        mask = runtime.Tensor.from_numpy(mask_np, device=device, queue=queue)
        prefix_sum = runtime.Tensor.from_numpy(prefix_np, device=device, queue=queue)
        output = runtime.Tensor.empty((3,), dtype="int32", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.compact_by_mask_i32(data, input_valid, mask, prefix_sum, output, output_valid, count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    np.testing.assert_array_equal(output_valid.to_numpy(), expected_valid)

def test_dataframe_coalesce_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "coalesce"):
        pytest.skip("no DataFrame coalesce binding in this build")
    col1_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    col2_np = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    col1_valid_np = np.array([0b00010001], dtype=np.uint32)
    col2_valid_np = np.array([0b00001110], dtype=np.uint32)
    expected = np.array([1.0, 20.0, 30.0, 40.0, 5.0], dtype=np.float32)
    expected_valid = np.array([0b00011111], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        col1 = runtime.Tensor.from_numpy(col1_np, device=device, queue=queue)
        col1_valid = runtime.Tensor.from_numpy(col1_valid_np, device=device, queue=queue)
        col2 = runtime.Tensor.from_numpy(col2_np, device=device, queue=queue)
        col2_valid = runtime.Tensor.from_numpy(col2_valid_np, device=device, queue=queue)
        output = runtime.Tensor.empty((5,), dtype="float32", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.coalesce(col1, col1_valid, col2, col2_valid, output, output_valid, count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)
    np.testing.assert_array_equal(output_valid.to_numpy(), expected_valid)

def test_dataframe_copy_if_valid_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "copy_if_valid"):
        pytest.skip("no DataFrame copy_if_valid binding in this build")
    src_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    dst_np = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
    src_valid_np = np.array([0b00011101], dtype=np.uint32)
    dst_valid_np = np.array([0b00010111], dtype=np.uint32)
    expected = np.array([1.0, 200.0, 3.0, 400.0, 5.0], dtype=np.float32)
    expected_valid = np.array([0b00010101], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        src = runtime.Tensor.from_numpy(src_np, device=device, queue=queue)
        src_valid = runtime.Tensor.from_numpy(src_valid_np, device=device, queue=queue)
        dst_valid = runtime.Tensor.from_numpy(dst_valid_np, device=device, queue=queue)
        dst = runtime.Tensor.from_numpy(dst_np, device=device, queue=queue)
        out_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.copy_if_valid(src, src_valid, dst_valid, dst, out_valid, count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)
    np.testing.assert_array_equal(out_valid.to_numpy(), expected_valid)

def test_dataframe_inclusive_scan_sum_valid_prefix_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "inclusive_scan_sum_valid_prefix"):
        pytest.skip("no DataFrame inclusive_scan_sum_valid_prefix binding in this build")
    data_np = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float32)
    input_valid_np = np.array([0b00011011], dtype=np.uint32)
    expected = np.array([1.0, 3.0, 3.0, 7.0, 12.0], dtype=np.float32)
    expected_valid = np.array([0b00000011], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        output = runtime.Tensor.empty((5,), dtype="float32", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.inclusive_scan_sum_valid_prefix(data, input_valid, output, output_valid, count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)
    np.testing.assert_array_equal(output_valid.to_numpy(), expected_valid)

def test_dataframe_inclusive_scan_sum_i32_to_i64_skip_nulls_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "inclusive_scan_sum_i32_to_i64_skip_nulls"):
        pytest.skip("no DataFrame inclusive_scan_sum_i32_to_i64_skip_nulls binding in this build")
    data_np = np.array([2147483647, 1, 5, -2, 10], dtype=np.int32)
    input_valid_np = np.array([0b00011011], dtype=np.uint32)
    expected = np.array([2147483647, 2147483648, 2147483648, 2147483646, 2147483656], dtype=np.int64)
    expected_valid = np.array([0b00011011], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        data = runtime.Tensor.from_numpy(data_np, device=device, queue=queue)
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        output = runtime.Tensor.empty((5,), dtype="int64", device=device, queue=queue)
        output_valid = runtime.Tensor.empty((1,), dtype="uint32", device=device, queue=queue)
        out = dataframe.inclusive_scan_sum_i32_to_i64_skip_nulls(data, input_valid, output, output_valid, count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    np.testing.assert_array_equal(output_valid.to_numpy(), expected_valid)

def test_dataframe_count_valid_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "count_valid"):
        pytest.skip("no DataFrame count_valid binding in this build")
    input_valid_np = np.array([0xFFFFFFB7], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        output = runtime.Tensor.empty((), dtype="uint32", device=device, queue=queue)
        out = dataframe.count_valid(input_valid, output, count=8, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), np.array(6, dtype=np.uint32))

def test_dataframe_is_valid_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "is_valid"):
        pytest.skip("no DataFrame is_valid binding in this build")
    input_valid_np = np.array([0xFFFFFFB7], dtype=np.uint32)
    expected = np.array([1, 1, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    with runtime.Queue(device=device) as queue:
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        output = runtime.Tensor.empty((8,), dtype="uint8", device=device, queue=queue)
        out = dataframe.is_valid(input_valid, output, count=8, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)

def test_dataframe_is_null_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "is_null"):
        pytest.skip("no DataFrame is_null binding in this build")
    input_valid_np = np.array([0xFFFFFFB7], dtype=np.uint32)
    expected = np.array([0, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8)
    with runtime.Queue(device=device) as queue:
        input_valid = runtime.Tensor.from_numpy(input_valid_np, device=device, queue=queue)
        output = runtime.Tensor.empty((8,), dtype="uint8", device=device, queue=queue)
        out = dataframe.is_null(input_valid, output, count=8, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)

def test_dataframe_fill_constant_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "fill_constant"):
        pytest.skip("no DataFrame fill_constant binding in this build")
    expected = np.full((5,), 2.5, dtype=np.float32)
    with runtime.Queue(device=device) as queue:
        output = runtime.Tensor.empty((5,), dtype="float32", device=device, queue=queue)
        out = dataframe.fill_constant(output, value=2.5, count=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)

def test_dataframe_fill_constant_i32_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "fill_constant_i32"):
        pytest.skip("no DataFrame fill_constant_i32 binding in this build")
    expected = np.array([-7, -7, -7, -7], dtype=np.int32)
    with runtime.Queue(device=device) as queue:
        output = runtime.Tensor.empty((4,), dtype="int32", device=device, queue=queue)
        out = dataframe.fill_constant_i32(output, value=-7, count=4, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)

def test_dataframe_fill_range_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "fill_range"):
        pytest.skip("no DataFrame fill_range binding in this build")
    expected = np.array([-2, -1, 0, 1, 2, 3], dtype=np.int32)
    with runtime.Queue(device=device) as queue:
        output = runtime.Tensor.empty((6,), dtype="int32", device=device, queue=queue)
        out = dataframe.fill_range(output, start=-2, count=6, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)

def test_dataframe_fill_validity_all_valid_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "fill_validity_all_valid"):
        pytest.skip("no DataFrame fill_validity_all_valid binding in this build")
    expected = np.array([0xFFFFFFFF, 0x00000007], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        output = runtime.Tensor.empty((2,), dtype="uint32", device=device, queue=queue)
        out = dataframe.fill_validity_all_valid(output, count=35, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)

def test_dataframe_fill_validity_all_null_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(dataframe, "fill_validity_all_null"):
        pytest.skip("no DataFrame fill_validity_all_null binding in this build")
    expected = np.array([0, 0], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        output = runtime.Tensor.empty((2,), dtype="uint32", device=device, queue=queue)
        out = dataframe.fill_validity_all_null(output, count=35, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)

def test_ml_mean_squared_error_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(ml, "mean_squared_error"):
        pytest.skip("no ML mean_squared_error binding in this build")
    y_true_np = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    y_pred_np = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    expected = np.mean((y_true_np - y_pred_np) ** 2).astype(np.float32)
    with runtime.Queue(device=device) as queue:
        y_true = runtime.Tensor.from_numpy(y_true_np, device=device, queue=queue)
        y_pred = runtime.Tensor.from_numpy(y_pred_np, device=device, queue=queue)
        result = runtime.Tensor.empty((), dtype="float32", device=device, queue=queue)
        out = ml.mean_squared_error(y_true, y_pred, result, count=3, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)

def test_ml_root_mean_squared_error_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(ml, "root_mean_squared_error"):
        pytest.skip("no ML root_mean_squared_error binding in this build")
    y_true_np = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    y_pred_np = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    expected = np.sqrt(np.mean((y_true_np - y_pred_np) ** 2)).astype(np.float32)
    with runtime.Queue(device=device) as queue:
        y_true = runtime.Tensor.from_numpy(y_true_np, device=device, queue=queue)
        y_pred = runtime.Tensor.from_numpy(y_pred_np, device=device, queue=queue)
        result = runtime.Tensor.empty((), dtype="float32", device=device, queue=queue)
        out = ml.root_mean_squared_error(y_true, y_pred, result, count=3, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)

def test_ml_mean_absolute_error_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(ml, "mean_absolute_error"):
        pytest.skip("no ML mean_absolute_error binding in this build")
    y_true_np = np.array([1.0, -2.0, 4.0], dtype=np.float32)
    y_pred_np = np.array([0.0, -1.0, 1.0], dtype=np.float32)
    expected = np.mean(np.abs(y_true_np - y_pred_np)).astype(np.float32)
    with runtime.Queue(device=device) as queue:
        y_true = runtime.Tensor.from_numpy(y_true_np, device=device, queue=queue)
        y_pred = runtime.Tensor.from_numpy(y_pred_np, device=device, queue=queue)
        result = runtime.Tensor.empty((), dtype="float32", device=device, queue=queue)
        out = ml.mean_absolute_error(y_true, y_pred, result, count=3, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)

def test_ml_mean_absolute_percentage_error_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(ml, "mean_absolute_percentage_error"):
        pytest.skip("no ML mean_absolute_percentage_error binding in this build")
    y_true_np = np.array([2.0, 4.0, 5.0], dtype=np.float32)
    y_pred_np = np.array([1.0, 6.0, 4.0], dtype=np.float32)
    expected = (100.0 * np.mean(np.abs(y_true_np - y_pred_np) / np.abs(y_true_np))).astype(np.float32)
    with runtime.Queue(device=device) as queue:
        y_true = runtime.Tensor.from_numpy(y_true_np, device=device, queue=queue)
        y_pred = runtime.Tensor.from_numpy(y_pred_np, device=device, queue=queue)
        result = runtime.Tensor.empty((), dtype="float32", device=device, queue=queue)
        out = ml.mean_absolute_percentage_error(y_true, y_pred, result, count=3, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)

def test_ml_r2_score_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(ml, "r2_score"):
        pytest.skip("no ML r2_score binding in this build")
    y_true_np = np.array([3.0, -0.5, 2.0, 7.0], dtype=np.float32)
    y_pred_np = np.array([2.5, 0.0, 2.0, 8.0], dtype=np.float32)
    y_mean_np = np.asarray(np.mean(y_true_np), dtype=np.float32)
    expected = (1.0 - np.sum((y_true_np - y_pred_np) ** 2) / np.sum((y_true_np - y_mean_np) ** 2)).astype(np.float32)
    with runtime.Queue(device=device) as queue:
        y_true = runtime.Tensor.from_numpy(y_true_np, device=device, queue=queue)
        y_pred = runtime.Tensor.from_numpy(y_pred_np, device=device, queue=queue)
        y_mean = runtime.Tensor.from_numpy(y_mean_np, device=device, queue=queue)
        result = runtime.Tensor.empty((), dtype="float32", device=device, queue=queue)
        out = ml.r2_score(y_true, y_pred, y_mean, result, count=4, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)

def test_ml_explained_variance_score_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(ml, "explained_variance_score"):
        pytest.skip("no ML explained_variance_score binding in this build")
    y_true_np = np.array([3.0, -0.5, 2.0, 7.0], dtype=np.float32)
    y_pred_np = np.array([2.5, 0.0, 2.0, 8.0], dtype=np.float32)
    residuals_np = y_true_np - y_pred_np
    mean_residuals_np = np.asarray(np.mean(residuals_np), dtype=np.float32)
    mean_true_np = np.asarray(np.mean(y_true_np), dtype=np.float32)
    expected = (1.0 - np.sum((residuals_np - mean_residuals_np) ** 2) / np.sum((y_true_np - mean_true_np) ** 2)).astype(np.float32)
    with runtime.Queue(device=device) as queue:
        y_true = runtime.Tensor.from_numpy(y_true_np, device=device, queue=queue)
        y_pred = runtime.Tensor.from_numpy(y_pred_np, device=device, queue=queue)
        mean_residuals = runtime.Tensor.from_numpy(mean_residuals_np, device=device, queue=queue)
        mean_true = runtime.Tensor.from_numpy(mean_true_np, device=device, queue=queue)
        result = runtime.Tensor.empty((), dtype="float32", device=device, queue=queue)
        out = ml.explained_variance_score(y_true, y_pred, mean_residuals, mean_true, result, count=4, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)

def test_sparse_csr_spmv_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(sparse, "csr_spmv"):
        pytest.skip("no CSR SpMV binding in this build")
    row_offsets_np = np.array([0, 2, 3, 5], dtype=np.uint32)
    col_indices_np = np.array([0, 2, 1, 0, 2], dtype=np.uint32)
    values_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    expected = np.array([70.0, 60.0, 190.0], dtype=np.float32)
    with runtime.Queue(device=device) as queue:
        row_offsets = runtime.Tensor.from_numpy(row_offsets_np, device=device, queue=queue)
        col_indices = runtime.Tensor.from_numpy(col_indices_np, device=device, queue=queue)
        values = runtime.Tensor.from_numpy(values_np, device=device, queue=queue)
        x = runtime.Tensor.from_numpy(x_np, device=device, queue=queue)
        y = runtime.Tensor.empty((3,), dtype="float32", device=device, queue=queue)
        out = sparse.csr_spmv(row_offsets, col_indices, values, x, y, rows=3, cols=3, nnz=5, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)

def test_sparse_csr_matrix_helper_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(sparse, "CSRMatrix"):
        pytest.skip("no CSRMatrix helper in this build")
    row_offsets_np = np.array([0, 2, 3, 5], dtype=np.uint32)
    col_indices_np = np.array([0, 2, 1, 0, 2], dtype=np.uint32)
    values_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    expected = np.array([70.0, 60.0, 190.0], dtype=np.float32)
    with runtime.Queue(device=device) as queue:
        matrix = sparse.CSRMatrix.from_numpy(row_offsets_np, col_indices_np, values_np, shape=(3, 3), device=device, queue=queue)
        x = runtime.Tensor.from_numpy(x_np, device=device, queue=queue)
        out = matrix.matvec(x, backend=device.backend, queue=queue, sync=True)
    assert matrix.shape == (3, 3)
    assert matrix.nnz == 5
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)

def test_graph_out_degree_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(graph, "out_degree"):
        pytest.skip("no graph out_degree binding in this build")
    row_offsets_np = np.array([0, 2, 3, 5], dtype=np.uint32)
    expected = np.array([2, 1, 2], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        row_offsets = runtime.Tensor.from_numpy(row_offsets_np, device=device, queue=queue)
        degrees = runtime.Tensor.empty((3,), dtype="uint32", device=device, queue=queue)
        out = graph.out_degree(row_offsets, degrees, vertex_count=3, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)

def test_graph_in_degree_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if not hasattr(graph, "in_degree"):
        pytest.skip("no graph in_degree binding in this build")
    col_offsets_np = np.array([0, 1, 4, 5], dtype=np.uint32)
    expected = np.array([1, 3, 1], dtype=np.uint32)
    with runtime.Queue(device=device) as queue:
        col_offsets = runtime.Tensor.from_numpy(col_offsets_np, device=device, queue=queue)
        degrees = runtime.Tensor.empty((3,), dtype="uint32", device=device, queue=queue)
        out = graph.in_degree(col_offsets, degrees, vertex_count=3, backend=device.backend, queue=queue, sync=True)
    np.testing.assert_array_equal(out.to_numpy(), expected)

def test_dlpack_surface_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    tensor = runtime.Tensor.from_numpy(np.arange(4, dtype=np.float32), device=device)
    dl_device = tensor.__dlpack_device__()
    assert isinstance(dl_device, tuple)
    assert len(dl_device) == 2
    caps = runtime.get_capabilities()
    assert "live_dlpack_exports" in caps
    for status in caps.get("backends", {}).values():
        memory = status.get("memory")
        assert isinstance(memory, dict)
        assert {"supported", "free_bytes", "total_bytes", "error_code"}.issubset(memory)
        assert memory["free_bytes"] >= 0
        assert memory["total_bytes"] >= 0
        if memory["supported"]:
            assert memory["total_bytes"] >= memory["free_bytes"]
    if device.backend not in {"cuda", "hip"}:
        with pytest.raises(RuntimeError, match="CUDA/HIP"):
            tensor.__dlpack__()
        tensor.close()
        assert getattr(tensor, "closed", getattr(tensor, "_closed", False))
    else:
        import gc
        for stream in (0, 1, 2):
            capsule = tensor.__dlpack__(stream=stream)
            assert type(capsule).__name__ == "PyCapsule"
            del capsule
            gc.collect()
        with pytest.raises(Exception):
            tensor.__dlpack__(stream=-1)
        with pytest.raises(TypeError):
            tensor.__dlpack__(stream=object())

def test_dlpack_imported_tensor_lifetime_retained_until_queue_sync():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if device.backend not in {"cuda", "hip"}:
        pytest.skip("DLPack zero-copy import is currently CUDA/HIP-only")
    src_np = np.array([3.0, 4.0, 12.0], dtype=np.float32)
    source = runtime.Tensor.from_numpy(src_np, device=device)
    imported = runtime.Tensor.from_dlpack(source.__dlpack__(stream=0), device=device)
    result = runtime.Tensor.empty((), dtype="float32", device=device)
    with runtime.Queue(device=device) as queue:
        linalg.nrm2(imported, result, backend=device.backend, queue=queue, sync=False)
        imported.close()
        with pytest.raises(runtime.RuntimeFailureError, match="live DLPack exports"):
            source.close()
        queue.sync()
    np.testing.assert_allclose(result.to_numpy(), np.linalg.norm(src_np).astype(np.float32), rtol=1e-6)
    source.close()

def test_dlpack_cross_backend_mismatch_rejected_without_copy():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if device.backend not in {"cuda", "hip"}:
        pytest.skip("DLPack zero-copy import is currently CUDA/HIP-only")
    opposite = "hip" if device.backend == "cuda" else "cuda"
    tensor = runtime.Tensor.from_numpy(np.arange(4, dtype=np.float32), device=device)
    capsule = tensor.__dlpack__(stream=0)
    with pytest.raises(RuntimeError, match="Cannot zero-copy DLPack tensor into requested backend"):
        runtime.Tensor.from_dlpack(capsule, device=runtime.Device(opposite))
    tensor.close()

def test_dlpack_exported_capsule_retains_tensor_until_consumed():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if device.backend not in {"cuda", "hip"}:
        pytest.skip("DLPack zero-copy export is currently CUDA/HIP-only")
    import gc
    tensor = runtime.Tensor.from_numpy(np.arange(4, dtype=np.float32), device=device)
    capsule = tensor.__dlpack__(stream=0)
    with pytest.raises(runtime.RuntimeFailureError, match="live DLPack exports"):
        tensor.close()
    del capsule
    gc.collect()
    tensor.close()

def test_dlpack_deleter_after_runtime_module_teardown_is_no_throw():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if device.backend not in {"cuda", "hip"}:
        pytest.skip("DLPack zero-copy export is currently CUDA/HIP-only")
    import gc
    import navatala_gpu._core as core
    tensor = runtime.Tensor.from_numpy(np.arange(4, dtype=np.float32), device=device)
    capsule = tensor.__dlpack__(stream=0)
    old_ext = getattr(core, "_ext", None)
    old_error = getattr(core, "_extension_error", None)
    try:
        core._ext = None
        core._extension_error = RuntimeError("simulated teardown")
        del capsule
        gc.collect()
    finally:
        core._ext = old_ext
        core._extension_error = old_error
    tensor.close()

def test_dlpack_export_roundtrip_telemetry_returns_to_baseline():
    np = pytest.importorskip("numpy")
    device = _available_device()
    if device.backend not in {"cuda", "hip"}:
        pytest.skip("DLPack zero-copy export is currently CUDA/HIP-only")
    import gc
    tensor = runtime.Tensor.from_numpy(np.arange(4, dtype=np.float32), device=device)
    baseline = runtime.get_capabilities().get("live_dlpack_exports")
    for _ in range(1000):
        capsule = tensor.__dlpack__(stream=0)
        del capsule
    gc.collect()
    assert runtime.get_capabilities().get("live_dlpack_exports") == baseline
    tensor.close()

def test_readonly_role_validation_accepts_inputs_and_rejects_outputs():
    from navatala_gpu.runtime import validate_call
    class FakeTensor:
        def __init__(self, *, readonly=False):
            self.shape = (3,)
            self.dtype = "float32"
            self.device = runtime.Device("cuda")
            self.readonly = readonly
    validate_call("linalg", "axpy", {"x": FakeTensor(readonly=True), "y": FakeTensor(readonly=False)}, {"alpha": 1.0, "backend": "cuda"})
    with pytest.raises(PermissionError, match="requires write access"):
        validate_call("linalg", "axpy", {"x": FakeTensor(readonly=False), "y": FakeTensor(readonly=True)}, {"alpha": 1.0, "backend": "cuda"})

def test_dlpack_backend_neutral_adapter_surface():
    assert runtime.dlpack_device_to_backend(2) == "cuda"
    assert runtime.dlpack_device_to_backend(10) == "hip"
    assert runtime.dlpack_backend_device_type("cuda") == 2
    assert runtime.dlpack_backend_device_type(runtime.Device("hip")) == 10
    assert runtime.can_zero_copy_dlpack("cuda", "cuda", dtype="float32")
    assert runtime.can_zero_copy_dlpack((2, 0), runtime.Device("cuda"))
    assert not runtime.can_zero_copy_dlpack("cuda", "hip")
    assert not runtime.can_zero_copy_dlpack("opencl", "opencl")
    adapter = runtime.DLPackAdapter(runtime.Device("cuda"))
    assert adapter.supports_zero_copy_import((2, 0), dtype="float32")
    assert adapter.supports_zero_copy_export("cuda", dtype="float32")
    assert not adapter.supports_zero_copy_import((10, 0), dtype="float32")
    with pytest.raises(runtime.UnsupportedBackendError):
        runtime.dlpack_device_to_backend(9999)

def test_internal_kernel_launch_fixture_if_backend_available():
    np = pytest.importorskip("numpy")
    device = _available_device()
    src_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    src = runtime.Tensor.from_numpy(src_np, device=device)
    dst = runtime.Tensor.empty_like(src)
    with runtime.Queue(device=device) as queue:
        try:
            kernels.launch(
                "__navatala_stub_fixture__",
                "__navatala_test_copy_f32",
                [src, dst],
                source_kind="opencl_c",
                queue=queue,
                grid=(1,),
                block=(1,),
                sync=True,
            )
        except (runtime.UnsupportedBackendError, runtime.GpuRuntimeError, runtime.RuntimeFailureError) as exc:
            pytest.skip(f"raw kernel launch fixture is not available for backend {device.backend}: {exc}")
    np.testing.assert_allclose(dst.to_numpy(), src_np, rtol=1e-6)
