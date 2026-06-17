// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <Python.h>
#include <dlpack/dlpack.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include "navatala/navatala_ffi.h"
#include "runtime_helpers.hpp"

namespace nb = nanobind;
static_assert(NAVATALA_GPU_FFI_ABI_VERSION == 6, "navatala_gpu bindings are incompatible with this navatala_ffi.h ABI version");

namespace navatala_gpu::python {

const char* ffi_error_name(NavatalaErrorCode code) {
    switch (code) {
        case NAVATALA_SUCCESS: return "success";
        case NAVATALA_INVALID_HANDLE: return "invalid_handle";
        case NAVATALA_INVALID_PARAM: return "invalid_param";
        case NAVATALA_GPU_ERROR: return "gpu_error";
        case NAVATALA_OUT_OF_MEMORY: return "out_of_memory";
        case NAVATALA_NOT_IMPLEMENTED: return "not_implemented";
        case NAVATALA_IO_ERROR: return "io_error";
        case NAVATALA_RUNTIME_ERROR: return "runtime_error";
        case NAVATALA_OVERFLOW_ERROR: return "overflow_error";
        case NAVATALA_TIMEOUT: return "timeout";
        case NAVATALA_CANCELLED: return "cancelled";
        case NAVATALA_NOT_FOUND: return "not_found";
        case NAVATALA_ALREADY_EXISTS: return "already_exists";
        case NAVATALA_PERMISSION_DENIED: return "permission_denied";
        case NAVATALA_INTERNAL_ERROR: return "internal_error";
        default: return "unknown_error";
    }
}

NavatalaProgramSourceKind program_source_kind_from_string(const std::string& value) {
    if (value == "cuda_cpp" || value == "cuda") return NAVATALA_PROGRAM_SOURCE_CUDA_CPP;
    if (value == "hip_cpp" || value == "hip") return NAVATALA_PROGRAM_SOURCE_HIP_CPP;
    if (value == "opencl_c" || value == "opencl") return NAVATALA_PROGRAM_SOURCE_OPENCL_C;
    if (value == "opencl_binary") return NAVATALA_PROGRAM_SOURCE_OPENCL_BINARY;
    if (value == "glsl") return NAVATALA_PROGRAM_SOURCE_GLSL;
    if (value == "spirv" || value == "spv") return NAVATALA_PROGRAM_SOURCE_SPIRV;
    if (value == "msl" || value == "metal") return NAVATALA_PROGRAM_SOURCE_MSL;
    if (value == "ptx") return NAVATALA_PROGRAM_SOURCE_PTX;
    if (value == "cubin") return NAVATALA_PROGRAM_SOURCE_CUBIN;
    if (value == "hsaco") return NAVATALA_PROGRAM_SOURCE_HSACO;
    if (value == "metallib") return NAVATALA_PROGRAM_SOURCE_METALLIB;
    throw StatusError(NAVATALA_INVALID_PARAM, "unsupported program source kind: " + value);
}

nb::object python_exception_for_code(NavatalaErrorCode code) {
    nb::object exceptions = nb::module_::import_("navatala_gpu._exceptions");
    switch (code) {
        case NAVATALA_INVALID_HANDLE: return exceptions.attr("InvalidHandleError");
        case NAVATALA_INVALID_PARAM: return exceptions.attr("InvalidArgumentError");
        case NAVATALA_GPU_ERROR: return exceptions.attr("GpuRuntimeError");
        case NAVATALA_OUT_OF_MEMORY: return exceptions.attr("OutOfMemoryError");
        case NAVATALA_NOT_IMPLEMENTED: return exceptions.attr("UnsupportedBackendError");
        case NAVATALA_IO_ERROR: return exceptions.attr("NavatalaIOError");
        case NAVATALA_RUNTIME_ERROR: return exceptions.attr("RuntimeFailureError");
        case NAVATALA_OVERFLOW_ERROR: return exceptions.attr("InvalidShapeError");
        case NAVATALA_TIMEOUT: return exceptions.attr("TimeoutGpuError");
        case NAVATALA_CANCELLED: return exceptions.attr("CancelledError");
        case NAVATALA_NOT_FOUND: return exceptions.attr("NotFoundError");
        case NAVATALA_ALREADY_EXISTS: return exceptions.attr("AlreadyExistsError");
        case NAVATALA_PERMISSION_DENIED: return exceptions.attr("PermissionDeniedError");
        case NAVATALA_INTERNAL_ERROR: return exceptions.attr("NavatalaInternalError");
        default: return exceptions.attr("RuntimeFailureError");
    }
}

[[noreturn]] void throw_python_error(NavatalaErrorCode code, const std::string& message) {
    nb::object exc = python_exception_for_code(code);
    PyErr_SetString(exc.ptr(), message.c_str());
    throw nb::python_error();
}

template <typename Fn>
auto translate_status(Fn&& fn, const char* operation) {
    try {
        return fn();
    } catch (const StatusError& err) {
        throw_python_error(err.code(), std::string(operation) + " failed: " + err.what());
    }
}

NavatalaBackend backend_from_device_object(nb::object device) {
    if (device.is_none()) {
        return NAVATALA_BACKEND_AUTO_FFI;
    }
    nb::object value = device.attr("backend");
    if (value.is_none()) {
        return NAVATALA_BACKEND_AUTO_FFI;
    }
    return parse_backend(nb::cast<std::string>(value));
}

int device_id_from_device_object(nb::object device) {
    if (device.is_none()) {
        return 0;
    }
    nb::object value = device.attr("ordinal");
    if (value.is_none()) {
        return 0;
    }
    return nb::cast<int>(value);
}

constexpr const char* kQueueCapsuleName = "navatala_gpu.queue";
constexpr const char* kUsedQueueCapsuleName = "used_navatala_gpu.queue";

void queue_capsule_destructor(PyObject* capsule) noexcept {
    if (capsule == nullptr) {
        return;
    }
    if (PyCapsule_IsValid(capsule, kQueueCapsuleName)) {
        auto* holder = static_cast<std::shared_ptr<QueueControlBlock>*>(PyCapsule_GetPointer(capsule, kQueueCapsuleName));
        delete holder;
        return;
    }
    PyErr_Clear();
}

std::shared_ptr<QueueControlBlock>& queue_control_from_capsule(nb::object capsule_obj) {
    if (!PyCapsule_CheckExact(capsule_obj.ptr())) {
        throw StatusError(NAVATALA_INVALID_PARAM, "queue handle is not a PyCapsule");
    }
    if (!PyCapsule_IsValid(capsule_obj.ptr(), kQueueCapsuleName)) {
        PyErr_Clear();
        throw StatusError(NAVATALA_INVALID_HANDLE, "queue is closed or invalid");
    }
    auto* holder = static_cast<std::shared_ptr<QueueControlBlock>*>(PyCapsule_GetPointer(capsule_obj.ptr(), kQueueCapsuleName));
    if (holder == nullptr || !*holder || (*holder)->handle == nullptr) {
        throw StatusError(NAVATALA_INVALID_HANDLE, "queue handle is empty");
    }
    return *holder;
}

nb::capsule make_queue_capsule(std::shared_ptr<QueueControlBlock> queue) {
    auto* holder = new std::shared_ptr<QueueControlBlock>(std::move(queue));
    PyObject* capsule = PyCapsule_New(holder, kQueueCapsuleName, &queue_capsule_destructor);
    if (capsule == nullptr) {
        delete holder;
        throw nb::python_error();
    }
    return nb::steal<nb::capsule>(capsule);
}

std::shared_ptr<QueueControlBlock> queue_control_ptr_from_object(nb::object queue) {
    if (queue.is_none()) {
        throw StatusError(NAVATALA_INVALID_PARAM, "queue is required");
    }
    nb::object handle = nb::none();
    if (PyCapsule_CheckExact(queue.ptr())) {
        handle = queue;
    } else {
        if (PyObject_HasAttrString(queue.ptr(), "_ensure_handle")) {
            queue.attr("_ensure_handle")();
        }
        if (PyObject_HasAttrString(queue.ptr(), "_handle")) {
            handle = queue.attr("_handle");
        }
    }
    if (handle.is_none()) {
        throw StatusError(NAVATALA_INVALID_HANDLE, "queue has no runtime handle");
    }
    return queue_control_from_capsule(handle);
}

nb::capsule create_queue_capsule(nb::object device, int priority) {
    auto context = std::make_shared<ContextControlBlock>(
        backend_from_device_object(device),
        device_id_from_device_object(device));
    auto queue = std::make_shared<QueueControlBlock>(context, priority);
    return make_queue_capsule(std::move(queue));
}

void close_queue_capsule(nb::object capsule_obj) {
    if (capsule_obj.is_none()) {
        return;
    }
    if (!PyCapsule_CheckExact(capsule_obj.ptr())) {
        throw StatusError(NAVATALA_INVALID_PARAM, "queue handle is not a PyCapsule");
    }
    if (PyCapsule_IsValid(capsule_obj.ptr(), kUsedQueueCapsuleName)) {
        return;
    }
    PyErr_Clear();
    auto* holder = static_cast<std::shared_ptr<QueueControlBlock>*>(PyCapsule_GetPointer(capsule_obj.ptr(), kQueueCapsuleName));
    if (holder == nullptr) {
        throw nb::python_error();
    }
    if (PyCapsule_SetName(capsule_obj.ptr(), kUsedQueueCapsuleName) != 0) {
        throw nb::python_error();
    }
    delete holder;
}

NavatalaGpuQueue* queue_handle_from_object(nb::object queue) {
    if (queue.is_none()) {
        return nullptr;
    }
    auto control = queue_control_ptr_from_object(queue);
    return control->handle;
}

void sync_queue_object(nb::object queue) {
    if (queue.is_none()) {
        return;
    }
    auto control = queue_control_ptr_from_object(queue);
    control->synchronize_and_release();
}

void retain_buffer_until_queue_sync(nb::object queue, const std::shared_ptr<BufferControlBlock>& buffer) {
    if (queue.is_none() || !buffer) {
        return;
    }
    auto control = queue_control_ptr_from_object(queue);
    control->retain_buffer(buffer);
}

void retain_tensor_until_queue_sync(nb::object queue, const Tensor& tensor) {
    retain_buffer_until_queue_sync(queue, tensor.buffer);
}

void retain_raw_buffer_until_queue_sync(nb::object queue, const Buffer& buffer) {
    retain_buffer_until_queue_sync(queue, buffer.buffer);
}

nb::object dlpack_stream_from_queue_object(nb::object queue) {
    if (queue.is_none()) {
        return nb::none();
    }
    auto control = queue_control_ptr_from_object(queue);
    if (control == nullptr || control->handle == nullptr) {
        throw StatusError(NAVATALA_INVALID_HANDLE, "queue has no runtime handle");
    }
    uintptr_t stream_handle = 0;
    std::uint8_t supported = 0;
    check_status(navatala_gpu_queue_native_handle(control->handle, &stream_handle, &supported), "navatala_gpu_queue_native_handle");
    if (supported == 0) {
        return nb::none();
    }
    if (stream_handle <= 2) {
        throw StatusError(NAVATALA_RUNTIME_ERROR, "backend returned a reserved DLPack stream sentinel as a native handle");
    }
    return nb::int_(stream_handle);
}

constexpr const char* kEventCapsuleName = "navatala_gpu.event";
constexpr const char* kUsedEventCapsuleName = "used_navatala_gpu.event";

void event_capsule_destructor(PyObject* capsule) noexcept {
    if (capsule == nullptr) {
        return;
    }
    if (PyCapsule_IsValid(capsule, kEventCapsuleName)) {
        auto* holder = static_cast<std::shared_ptr<EventControlBlock>*>(PyCapsule_GetPointer(capsule, kEventCapsuleName));
        delete holder;
        return;
    }
    PyErr_Clear();
}

std::shared_ptr<EventControlBlock>& event_control_from_capsule(nb::object capsule_obj) {
    if (!PyCapsule_CheckExact(capsule_obj.ptr())) {
        throw StatusError(NAVATALA_INVALID_PARAM, "event handle is not a PyCapsule");
    }
    if (!PyCapsule_IsValid(capsule_obj.ptr(), kEventCapsuleName)) {
        PyErr_Clear();
        throw StatusError(NAVATALA_INVALID_HANDLE, "event is closed or invalid");
    }
    auto* holder = static_cast<std::shared_ptr<EventControlBlock>*>(PyCapsule_GetPointer(capsule_obj.ptr(), kEventCapsuleName));
    if (holder == nullptr || !*holder || (*holder)->handle == nullptr) {
        throw StatusError(NAVATALA_INVALID_HANDLE, "event handle is empty");
    }
    return *holder;
}

nb::capsule make_event_capsule(std::shared_ptr<EventControlBlock> event) {
    auto* holder = new std::shared_ptr<EventControlBlock>(std::move(event));
    PyObject* capsule = PyCapsule_New(holder, kEventCapsuleName, &event_capsule_destructor);
    if (capsule == nullptr) {
        delete holder;
        throw nb::python_error();
    }
    return nb::steal<nb::capsule>(capsule);
}

nb::capsule create_event_capsule(nb::object device) {
    auto context = std::make_shared<ContextControlBlock>(
        backend_from_device_object(device),
        device_id_from_device_object(device));
    auto event = std::make_shared<EventControlBlock>(context);
    return make_event_capsule(std::move(event));
}

nb::capsule create_event_capsule_from_queue(nb::object queue) {
    auto queue_control = queue_control_ptr_from_object(queue);
    auto event = std::make_shared<EventControlBlock>(queue_control->context);
    return make_event_capsule(std::move(event));
}

void close_event_capsule(nb::object capsule_obj) {
    if (capsule_obj.is_none()) {
        return;
    }
    if (!PyCapsule_CheckExact(capsule_obj.ptr())) {
        throw StatusError(NAVATALA_INVALID_PARAM, "event handle is not a PyCapsule");
    }
    if (PyCapsule_IsValid(capsule_obj.ptr(), kUsedEventCapsuleName)) {
        return;
    }
    PyErr_Clear();
    auto* holder = static_cast<std::shared_ptr<EventControlBlock>*>(PyCapsule_GetPointer(capsule_obj.ptr(), kEventCapsuleName));
    if (holder == nullptr) {
        throw nb::python_error();
    }
    if (PyCapsule_SetName(capsule_obj.ptr(), kUsedEventCapsuleName) != 0) {
        throw nb::python_error();
    }
    delete holder;
}

std::shared_ptr<EventControlBlock> event_control_ptr_from_object(nb::object event) {
    if (event.is_none()) {
        throw StatusError(NAVATALA_INVALID_PARAM, "event is required");
    }
    nb::object handle = nb::none();
    if (PyCapsule_CheckExact(event.ptr())) {
        handle = event;
    } else {
        if (PyObject_HasAttrString(event.ptr(), "_ensure_handle")) {
            event.attr("_ensure_handle")();
        }
        if (PyObject_HasAttrString(event.ptr(), "_handle")) {
            handle = event.attr("_handle");
        }
    }
    if (handle.is_none()) {
        throw StatusError(NAVATALA_INVALID_HANDLE, "event has no runtime handle");
    }
    return event_control_from_capsule(handle);
}

NavatalaGpuEvent* event_handle_from_object(nb::object event) {
    auto control = event_control_ptr_from_object(event);
    return control->handle;
}

void record_event_object(nb::object event, nb::object queue) {
    NavatalaGpuEvent* event_handle = event_handle_from_object(event);
    NavatalaGpuQueue* queue_handle = queue_handle_from_object(queue);
    if (event_handle == nullptr || queue_handle == nullptr) {
        throw StatusError(NAVATALA_INVALID_HANDLE, "event record requires valid event and queue");
    }
    check_status(navatala_gpu_event_record(event_handle, queue_handle), "navatala_gpu_event_record");
}

void wait_event_object(nb::object queue, nb::object event) {
    NavatalaGpuQueue* queue_handle = queue_handle_from_object(queue);
    NavatalaGpuEvent* event_handle = event_handle_from_object(event);
    if (queue_handle == nullptr || event_handle == nullptr) {
        throw StatusError(NAVATALA_INVALID_HANDLE, "event wait requires valid queue and event");
    }
    check_status(navatala_gpu_event_wait(queue_handle, event_handle), "navatala_gpu_event_wait");
}

void sync_event_object(nb::object event) {
    NavatalaGpuEvent* event_handle = event_handle_from_object(event);
    check_status(navatala_gpu_event_synchronize(event_handle), "navatala_gpu_event_synchronize");
}

bool event_is_complete_object(nb::object event) {
    NavatalaGpuEvent* event_handle = event_handle_from_object(event);
    return navatala_gpu_event_is_complete(event_handle) != 0;
}

std::vector<std::uint8_t> source_bytes_from_python(nb::object source) {
    if (PyBytes_Check(source.ptr())) {
        char* data = nullptr;
        Py_ssize_t size = 0;
        if (PyBytes_AsStringAndSize(source.ptr(), &data, &size) != 0) {
            throw nb::python_error();
        }
        return std::vector<std::uint8_t>(reinterpret_cast<std::uint8_t*>(data), reinterpret_cast<std::uint8_t*>(data) + size);
    }
    if (PyByteArray_Check(source.ptr())) {
        char* data = PyByteArray_AsString(source.ptr());
        Py_ssize_t size = PyByteArray_Size(source.ptr());
        if (data == nullptr || size < 0) {
            throw nb::python_error();
        }
        return std::vector<std::uint8_t>(reinterpret_cast<std::uint8_t*>(data), reinterpret_cast<std::uint8_t*>(data) + size);
    }
    if (PyUnicode_Check(source.ptr())) {
        Py_ssize_t size = 0;
        const char* data = PyUnicode_AsUTF8AndSize(source.ptr(), &size);
        if (data == nullptr || size < 0) {
            throw nb::python_error();
        }
        return std::vector<std::uint8_t>(reinterpret_cast<const std::uint8_t*>(data), reinterpret_cast<const std::uint8_t*>(data) + size);
    }
    throw StatusError(NAVATALA_INVALID_PARAM, "kernel source must be str, bytes, or bytearray");
}

std::array<std::uint32_t, 3> launch_triple_from_vector(const std::vector<std::uint32_t>& dims, const char* name) {
    if (dims.empty() || dims.size() > 3) {
        throw StatusError(NAVATALA_INVALID_PARAM, std::string(name) + " must have one to three positive dimensions");
    }
    std::array<std::uint32_t, 3> out{1, 1, 1};
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == 0) {
            throw StatusError(NAVATALA_INVALID_PARAM, std::string(name) + " dimensions must be positive");
        }
        out[i] = dims[i];
    }
    return out;
}

std::vector<NavatalaGpuBuffer*> buffer_handles_from_sequence(nb::sequence args) {
    std::vector<NavatalaGpuBuffer*> handles;
    handles.reserve(static_cast<std::size_t>(nb::len(args)));
    for (nb::handle item : args) {
        if (nb::isinstance<Tensor>(item)) {
            Tensor& tensor = nb::cast<Tensor&>(item);
            if (tensor.buffer == nullptr || tensor.buffer->handle == nullptr) {
                throw StatusError(NAVATALA_INVALID_HANDLE, "kernel tensor argument is closed");
            }
            handles.push_back(tensor.buffer->handle);
        } else if (nb::isinstance<Buffer>(item)) {
            Buffer& buffer = nb::cast<Buffer&>(item);
            if (buffer.buffer == nullptr || buffer.buffer->handle == nullptr) {
                throw StatusError(NAVATALA_INVALID_HANDLE, "kernel buffer argument is closed");
            }
            handles.push_back(buffer.buffer->handle);
        } else {
            throw StatusError(NAVATALA_INVALID_PARAM, "kernel launch arguments must be navatala_gpu Tensor or Buffer objects");
        }
    }
    return handles;
}

void retain_kernel_args_until_queue_sync(nb::object queue, nb::sequence args) {
    if (queue.is_none()) {
        return;
    }
    for (nb::handle item : args) {
        if (nb::isinstance<Tensor>(item)) {
            Tensor& tensor = nb::cast<Tensor&>(item);
            retain_tensor_until_queue_sync(queue, tensor);
        } else if (nb::isinstance<Buffer>(item)) {
            Buffer& buffer = nb::cast<Buffer&>(item);
            retain_raw_buffer_until_queue_sync(queue, buffer);
        }
    }
}

void launch_kernel_object(nb::object source, const std::string& entry_point, nb::sequence args, const std::string& source_kind, nb::object queue, std::vector<std::uint32_t> grid, std::vector<std::uint32_t> block, bool sync) {
    if (entry_point.empty()) {
        throw StatusError(NAVATALA_INVALID_PARAM, "kernel entry_point must not be empty");
    }
    auto queue_control = queue_control_ptr_from_object(queue);
    if (queue_control == nullptr || queue_control->context == nullptr || queue_control->context->handle == nullptr || queue_control->handle == nullptr) {
        throw StatusError(NAVATALA_INVALID_HANDLE, "kernel launch requires a live Queue");
    }
    std::vector<std::uint8_t> bytes = source_bytes_from_python(source);
    if (bytes.empty()) {
        throw StatusError(NAVATALA_INVALID_PARAM, "kernel source must not be empty");
    }
    auto grid_dims = launch_triple_from_vector(grid, "grid");
    auto block_dims = launch_triple_from_vector(block, "block");
    std::vector<NavatalaGpuBuffer*> buffers = buffer_handles_from_sequence(args);
    NavatalaGpuProgram* raw_program = nullptr;
    check_status(navatala_gpu_create_program(
        queue_control->context->handle,
        program_source_kind_from_string(source_kind),
        bytes.data(),
        bytes.size(),
        entry_point.c_str(),
        &raw_program),
        "navatala_gpu_create_program");
    std::unique_ptr<NavatalaGpuProgram, decltype(&navatala_gpu_destroy_program)> program(raw_program, &navatala_gpu_destroy_program);
    check_status(navatala_gpu_launch_program(
        queue_control->handle,
        program.get(),
        buffers.empty() ? nullptr : buffers.data(),
        buffers.size(),
        grid_dims[0], grid_dims[1], grid_dims[2],
        block_dims[0], block_dims[1], block_dims[2]),
        "navatala_gpu_launch_program");
    retain_kernel_args_until_queue_sync(queue, args);
    if (sync) {
        sync_queue_object(queue);
    }
}

std::atomic<std::uint64_t>& live_dlpack_export_count();

nb::dict backend_status(NavatalaBackend backend) {
    bool available = navatala_is_backend_available(backend) != 0;
    nb::dict status;
    status["compiled"] = available;
    status["available"] = available;
    status["initialized"] = available;
    status["selected"] = navatala_get_current_backend() == backend;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::uint8_t memory_supported = 0;
    NavatalaErrorCode memory_status = navatala_get_backend_memory_info(backend, 0, &free_bytes, &total_bytes, &memory_supported);
    nb::dict memory;
    memory["supported"] = memory_status == NAVATALA_SUCCESS && memory_supported != 0;
    memory["free_bytes"] = free_bytes;
    memory["total_bytes"] = total_bytes;
    memory["error_code"] = ffi_error_name(memory_status);
    status["memory"] = memory;
    return status;
}

nb::dict get_capabilities_dict() {
    nb::dict backends;
    backends["cuda"] = backend_status(NAVATALA_BACKEND_CUDA_FFI);
    backends["hip"] = backend_status(NAVATALA_BACKEND_HIP_FFI);
    backends["metal"] = backend_status(NAVATALA_BACKEND_METAL_FFI);
    backends["opencl"] = backend_status(NAVATALA_BACKEND_OPENCL_FFI);
    backends["vulkan"] = backend_status(NAVATALA_BACKEND_VULKAN_FFI);
    nb::dict caps;
    caps["extension_loaded"] = true;
    caps["ffi_abi_version"] = NAVATALA_GPU_FFI_ABI_VERSION;
    caps["backends"] = backends;
    caps["live_dlpack_exports"] = live_dlpack_export_count().load(std::memory_order_relaxed);
    return caps;
}

std::string dtype_from_buffer_format(const char* raw_format, Py_ssize_t itemsize) {
    std::string format = raw_format == nullptr ? "" : std::string(raw_format);
    if (format == "f" && itemsize == 4) return "float32";
    if (format == "d" && itemsize == 8) return "float64";
    if (format == "e" && itemsize == 2) return "float16";
    if ((format == "b" || format == "c") && itemsize == 1) return "int8";
    if (format == "B" && itemsize == 1) return "uint8";
    if (format == "h" && itemsize == 2) return "int16";
    if (format == "H" && itemsize == 2) return "uint16";
    if (format == "i" && itemsize == 4) return "int32";
    if (format == "I" && itemsize == 4) return "uint32";
    if (format == "q" && itemsize == 8) return "int64";
    if (format == "Q" && itemsize == 8) return "uint64";
    throw StatusError(NAVATALA_INVALID_PARAM, "unsupported Python buffer dtype format: " + format);
}

std::vector<std::uint64_t> shape_from_buffer(const Py_buffer& view) {
    std::vector<std::uint64_t> shape;
    if (view.ndim < 0) {
        throw StatusError(NAVATALA_INVALID_PARAM, "Python buffer has invalid negative rank");
    }
    if (view.ndim == 0) {
        if (view.itemsize <= 0) {
            throw StatusError(NAVATALA_INVALID_PARAM, "Python buffer has invalid itemsize");
        }
        if (view.len != view.itemsize) {
            throw StatusError(NAVATALA_INVALID_PARAM, "scalar Python buffer has inconsistent byte length");
        }
        return shape;
    }
    if (view.shape == nullptr) {
        throw StatusError(NAVATALA_INVALID_PARAM, "ranked Python buffer is missing shape metadata");
    }
    for (int i = 0; i < view.ndim; ++i) {
        if (view.shape[i] < 0) {
            throw StatusError(NAVATALA_INVALID_PARAM, "Python buffer has negative shape dimension");
        }
        shape.push_back(static_cast<std::uint64_t>(view.shape[i]));
    }
    return shape;
}

Tensor tensor_from_numpy_like(nb::object array, nb::object device, nb::object queue) {
    Py_buffer view;
    if (PyObject_GetBuffer(array.ptr(), &view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) != 0) {
        throw nb::python_error();
    }
    try {
        std::string dtype = dtype_from_buffer_format(view.format, view.itemsize);
        Tensor tensor = allocate_tensor(
            shape_from_buffer(view), dtype,
            backend_from_device_object(device),
            device_id_from_device_object(device));
        if (tensor.buffer == nullptr || tensor.buffer->handle == nullptr) {
            throw StatusError(NAVATALA_INVALID_HANDLE, "allocated tensor has no buffer handle");
        }
        check_status(navatala_gpu_copy_h2d(tensor.buffer->handle, view.buf, static_cast<std::size_t>(view.len), queue_handle_from_object(queue)), "navatala_gpu_copy_h2d");
        PyBuffer_Release(&view);
        return tensor;
    } catch (...) {
        PyBuffer_Release(&view);
        throw;
    }
}

nb::object tensor_to_numpy_array(Tensor& tensor, nb::object queue) {
    if (tensor.buffer == nullptr || tensor.buffer->handle == nullptr) {
        throw StatusError(NAVATALA_INVALID_HANDLE, "tensor is closed");
    }
    nb::object np = nb::module_::import_("numpy");
    nb::object array = np.attr("empty")(tensor.shape(), tensor.dtype());
    Py_buffer view;
    if (PyObject_GetBuffer(array.ptr(), &view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) != 0) {
        throw nb::python_error();
    }
    try {
        if (static_cast<std::size_t>(view.len) < tensor.nbytes()) {
            throw StatusError(NAVATALA_INTERNAL_ERROR, "NumPy output buffer is smaller than tensor");
        }
        check_status(navatala_gpu_copy_d2h(tensor.buffer->handle, view.buf, tensor.nbytes(), queue_handle_from_object(queue)), "navatala_gpu_copy_d2h");
        PyBuffer_Release(&view);
        return array;
    } catch (...) {
        PyBuffer_Release(&view);
        throw;
    }
}

[[noreturn]] void raise_python_value_error(const std::string& message) {
    PyErr_SetString(PyExc_ValueError, message.c_str());
    throw nb::python_error();
}

[[noreturn]] void raise_python_type_error(const std::string& message) {
    PyErr_SetString(PyExc_TypeError, message.c_str());
    throw nb::python_error();
}

[[noreturn]] void raise_python_runtime_error(const std::string& message) {
    PyErr_SetString(PyExc_RuntimeError, message.c_str());
    throw nb::python_error();
}

DLDeviceType dlpack_device_type(NavatalaBackend backend) {
    switch (backend) {
        case NAVATALA_BACKEND_CUDA_FFI: return kDLCUDA;
        case NAVATALA_BACKEND_HIP_FFI: return kDLROCM;
        case NAVATALA_BACKEND_METAL_FFI: return kDLMetal;
        case NAVATALA_BACKEND_OPENCL_FFI: return kDLOpenCL;
        case NAVATALA_BACKEND_VULKAN_FFI: return kDLVulkan;
        default: throw StatusError(NAVATALA_INVALID_PARAM, "DLPack export requires a resolved backend");
    }
}

NavatalaBackend backend_from_dlpack_device(DLDeviceType device_type) {
    switch (device_type) {
        case kDLCUDA: return NAVATALA_BACKEND_CUDA_FFI;
        case kDLROCM: return NAVATALA_BACKEND_HIP_FFI;
        case kDLMetal: return NAVATALA_BACKEND_METAL_FFI;
        case kDLOpenCL: return NAVATALA_BACKEND_OPENCL_FFI;
        case kDLVulkan: return NAVATALA_BACKEND_VULKAN_FFI;
        default: throw StatusError(NAVATALA_NOT_IMPLEMENTED, "navatala_gpu cannot consume DLPack tensor on unsupported device type");
    }
}

DLDataType dlpack_dtype_from_string(const std::string& dtype) {
    if (dtype == "float16" || dtype == "Float16") return DLDataType{static_cast<std::uint8_t>(kDLFloat), 16, 1};
    if (dtype == "float32" || dtype == "Float32") return DLDataType{static_cast<std::uint8_t>(kDLFloat), 32, 1};
    if (dtype == "float64" || dtype == "Float64") return DLDataType{static_cast<std::uint8_t>(kDLFloat), 64, 1};
    if (dtype == "bfloat16" || dtype == "BFloat16" || dtype == "bf16") return DLDataType{static_cast<std::uint8_t>(kDLBfloat), 16, 1};
    if (dtype == "int8" || dtype == "Int8") return DLDataType{static_cast<std::uint8_t>(kDLInt), 8, 1};
    if (dtype == "int16" || dtype == "Int16") return DLDataType{static_cast<std::uint8_t>(kDLInt), 16, 1};
    if (dtype == "int32" || dtype == "Int32") return DLDataType{static_cast<std::uint8_t>(kDLInt), 32, 1};
    if (dtype == "int64" || dtype == "Int64") return DLDataType{static_cast<std::uint8_t>(kDLInt), 64, 1};
    if (dtype == "uint8" || dtype == "UInt8") return DLDataType{static_cast<std::uint8_t>(kDLUInt), 8, 1};
    if (dtype == "uint16" || dtype == "UInt16") return DLDataType{static_cast<std::uint8_t>(kDLUInt), 16, 1};
    if (dtype == "uint32" || dtype == "UInt32") return DLDataType{static_cast<std::uint8_t>(kDLUInt), 32, 1};
    if (dtype == "uint64" || dtype == "UInt64") return DLDataType{static_cast<std::uint8_t>(kDLUInt), 64, 1};
    throw StatusError(NAVATALA_INVALID_PARAM, "navatala_gpu does not support DLPack dtype " + dtype);
}

std::string dtype_from_dlpack(DLDataType dtype) {
    if (dtype.lanes != 1) {
        raise_python_type_error("navatala_gpu does not support DLPack vector lanes");
    }
    if (dtype.code == static_cast<std::uint8_t>(kDLFloat) && dtype.bits == 16) return "float16";
    if (dtype.code == static_cast<std::uint8_t>(kDLFloat) && dtype.bits == 32) return "float32";
    if (dtype.code == static_cast<std::uint8_t>(kDLFloat) && dtype.bits == 64) return "float64";
    if (dtype.code == static_cast<std::uint8_t>(kDLBfloat) && dtype.bits == 16) return "bfloat16";
    if (dtype.code == static_cast<std::uint8_t>(kDLInt) && dtype.bits == 8) return "int8";
    if (dtype.code == static_cast<std::uint8_t>(kDLInt) && dtype.bits == 16) return "int16";
    if (dtype.code == static_cast<std::uint8_t>(kDLInt) && dtype.bits == 32) return "int32";
    if (dtype.code == static_cast<std::uint8_t>(kDLInt) && dtype.bits == 64) return "int64";
    if (dtype.code == static_cast<std::uint8_t>(kDLUInt) && dtype.bits == 8) return "uint8";
    if (dtype.code == static_cast<std::uint8_t>(kDLUInt) && dtype.bits == 16) return "uint16";
    if (dtype.code == static_cast<std::uint8_t>(kDLUInt) && dtype.bits == 32) return "uint32";
    if (dtype.code == static_cast<std::uint8_t>(kDLUInt) && dtype.bits == 64) return "uint64";
    raise_python_type_error("navatala_gpu does not support DLPack dtype code=" + std::to_string(dtype.code) + ", bits=" + std::to_string(dtype.bits) + ", lanes=" + std::to_string(dtype.lanes));
}

std::vector<std::uint64_t> shape_from_dlpack_tensor(const DLTensor& dl_tensor) {
    if (dl_tensor.ndim < 0 || (dl_tensor.ndim > 0 && dl_tensor.shape == nullptr)) {
        raise_python_value_error("DLPack tensor has invalid shape metadata");
    }
    std::vector<std::uint64_t> shape;
    for (int32_t i = 0; i < dl_tensor.ndim; ++i) {
        if (dl_tensor.shape[i] < 0) {
            raise_python_value_error("DLPack tensor has negative shape dimension");
        }
        shape.push_back(static_cast<std::uint64_t>(dl_tensor.shape[i]));
    }
    return shape;
}

void validate_compact_row_major(const DLTensor& dl_tensor) {
    if (dl_tensor.strides == nullptr) {
        return;
    }
    std::int64_t expected = 1;
    for (int32_t i = dl_tensor.ndim - 1; i >= 0; --i) {
        if (dl_tensor.strides[i] != expected) {
            raise_python_value_error("navatala_gpu currently requires contiguous row-major DLPack tensors");
        }
        expected *= dl_tensor.shape[i];
    }
}

std::size_t dlpack_tensor_nbytes(const DLTensor& dl_tensor) {
    std::vector<std::uint64_t> shape = shape_from_dlpack_tensor(dl_tensor);
    std::uint64_t elements = checked_element_count(shape);
    std::size_t itemsize = (static_cast<std::size_t>(dl_tensor.dtype.bits) * dl_tensor.dtype.lanes + 7) / 8;
    if (elements > std::numeric_limits<std::size_t>::max() / itemsize) {
        throw StatusError(NAVATALA_OVERFLOW_ERROR, "DLPack tensor byte size overflows size_t");
    }
    return static_cast<std::size_t>(elements) * itemsize;
}

struct DlpackExportContext {
    std::shared_ptr<BufferControlBlock> buffer;
    std::vector<std::int64_t> shape;
};

std::atomic<std::uint64_t>& live_dlpack_export_count() {
    static std::atomic<std::uint64_t> count{0};
    return count;
}

void release_dlpack_export_context(DlpackExportContext* ctx) noexcept {
    if (ctx == nullptr) {
        return;
    }
    if (ctx->buffer != nullptr) {
        ctx->buffer->live_dlpack_exports.fetch_sub(1, std::memory_order_release);
    }
    live_dlpack_export_count().fetch_sub(1, std::memory_order_relaxed);
    delete ctx;
}

void dlpack_legacy_export_deleter(DLManagedTensor* self) noexcept {
    if (self == nullptr) {
        return;
    }
    try {
        release_dlpack_export_context(static_cast<DlpackExportContext*>(self->manager_ctx));
        delete self;
    } catch (...) {
    }
}

void dlpack_versioned_export_deleter(DLManagedTensorVersioned* self) noexcept {
    if (self == nullptr) {
        return;
    }
    try {
        release_dlpack_export_context(static_cast<DlpackExportContext*>(self->manager_ctx));
        delete self;
    } catch (...) {
    }
}

void dlpack_capsule_destructor(PyObject* capsule) noexcept {
    if (capsule == nullptr) {
        return;
    }
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto* self = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
        if (self != nullptr && self->deleter != nullptr) self->deleter(self);
        return;
    }
    if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
        auto* self = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(capsule, "dltensor_versioned"));
        if (self != nullptr && self->deleter != nullptr) self->deleter(self);
        return;
    }
    PyErr_Clear();
}

nb::capsule make_dlpack_capsule(void* ptr, const char* name) {
    PyObject* capsule = PyCapsule_New(ptr, name, &dlpack_capsule_destructor);
    if (capsule == nullptr) {
        throw nb::python_error();
    }
    return nb::steal<nb::capsule>(capsule);
}

DlpackExportContext* make_dlpack_export_context(const Tensor& tensor) {
    auto* ctx = new DlpackExportContext;
    ctx->buffer = tensor.buffer;
    ctx->shape.reserve(tensor.shape_values.size());
    for (std::uint64_t dim : tensor.shape_values) {
        if (dim > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            delete ctx;
            throw StatusError(NAVATALA_OVERFLOW_ERROR, "DLPack shape dimension overflows int64");
        }
        ctx->shape.push_back(static_cast<std::int64_t>(dim));
    }
    if (ctx->buffer != nullptr) {
        ctx->buffer->live_dlpack_exports.fetch_add(1, std::memory_order_release);
    }
    live_dlpack_export_count().fetch_add(1, std::memory_order_relaxed);
    return ctx;
}

void fill_dlpack_tensor(DLTensor& dl_tensor, const Tensor& tensor, DlpackExportContext* ctx) {
    if (tensor.buffer == nullptr || tensor.buffer->handle == nullptr) {
        throw StatusError(NAVATALA_INVALID_HANDLE, "tensor is closed");
    }
    NavatalaBackend backend = tensor.buffer->context->backend;
    if (backend != NAVATALA_BACKEND_CUDA_FFI && backend != NAVATALA_BACKEND_HIP_FFI) {
        raise_python_runtime_error("DLPack zero-copy export is currently implemented for CUDA/HIP tensors only; use Tensor.to_numpy() for explicit copies");
    }
    dl_tensor.data = navatala_gpu_buffer_ptr(tensor.buffer->handle);
    dl_tensor.device = DLDevice{dlpack_device_type(backend), tensor.buffer->context->device_id};
    dl_tensor.ndim = static_cast<int32_t>(ctx->shape.size());
    dl_tensor.dtype = dlpack_dtype_from_string(tensor.dtype_value);
    dl_tensor.shape = ctx->shape.data();
    dl_tensor.strides = nullptr;
    dl_tensor.byte_offset = 0;
}

bool requested_versioned_dlpack(nb::object max_version) {
    if (max_version.is_none()) {
        return false;
    }
    int major = -1;
    if (PyLong_Check(max_version.ptr())) {
        major = static_cast<int>(PyLong_AsLong(max_version.ptr()));
    } else if (PyTuple_Check(max_version.ptr()) || PyList_Check(max_version.ptr())) {
        Py_ssize_t size = PyTuple_Check(max_version.ptr()) ? PyTuple_Size(max_version.ptr()) : PyList_Size(max_version.ptr());
        if (size < 1) {
            raise_python_value_error("Malformed DLPack max_version");
        }
        PyObject* item = PyTuple_Check(max_version.ptr()) ? PyTuple_GetItem(max_version.ptr(), 0) : PyList_GetItem(max_version.ptr(), 0);
        major = static_cast<int>(PyLong_AsLong(item));
    } else {
        raise_python_value_error("Malformed DLPack max_version");
    }
    if (major < 1) {
        raise_python_value_error("navatala_gpu requires DLPack max_version >= 1 for versioned export");
    }
    return true;
}

std::uintptr_t parse_dlpack_stream_token(nb::object stream) {
    if (stream.is_none()) {
        return 0;
    }
    if (!PyLong_Check(stream.ptr())) {
        raise_python_type_error("DLPack stream must be None or a non-negative integer token");
    }
    unsigned long long raw = PyLong_AsUnsignedLongLong(stream.ptr());
    if (PyErr_Occurred()) {
        throw nb::python_error();
    }
    if (raw > static_cast<unsigned long long>(std::numeric_limits<std::uintptr_t>::max())) {
        raise_python_value_error("DLPack stream token overflows uintptr_t");
    }
    return static_cast<std::uintptr_t>(raw);
}

void validate_dlpack_export_stream(const Tensor& tensor, nb::object stream) {
    std::uintptr_t token = parse_dlpack_stream_token(stream);
    if (token == 0) {
        return;
    }
    if (tensor.buffer == nullptr || tensor.buffer->context == nullptr) {
        throw StatusError(NAVATALA_INVALID_HANDLE, "tensor is closed");
    }
    NavatalaBackend backend = tensor.buffer->context->backend;
    if (backend != NAVATALA_BACKEND_CUDA_FFI && backend != NAVATALA_BACKEND_HIP_FFI) {
        raise_python_runtime_error("DLPack stream handoff is currently supported only for CUDA/HIP tensors; pass stream=None or use explicit copy");
    }
    // DLPack reserves stream tokens 1 and 2 as sentinels. They are accepted
    // here as tokens and deliberately never cast to native stream pointers.
}

nb::capsule tensor_to_dlpack_capsule(Tensor& tensor, nb::object stream, nb::object max_version) {
    validate_dlpack_export_stream(tensor, stream);
    if (requested_versioned_dlpack(max_version)) {
        auto* managed = new DLManagedTensorVersioned;
        auto* ctx = make_dlpack_export_context(tensor);
        managed->version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        managed->manager_ctx = ctx;
        managed->deleter = &dlpack_versioned_export_deleter;
        managed->flags = tensor.readonly ? DLPACK_FLAG_BITMASK_READ_ONLY : 0;
        try {
            fill_dlpack_tensor(managed->dl_tensor, tensor, ctx);
        } catch (...) {
            dlpack_versioned_export_deleter(managed);
            throw;
        }
        return make_dlpack_capsule(managed, "dltensor_versioned");
    }
    auto* managed = new DLManagedTensor;
    auto* ctx = make_dlpack_export_context(tensor);
    managed->manager_ctx = ctx;
    managed->deleter = &dlpack_legacy_export_deleter;
    try {
        fill_dlpack_tensor(managed->dl_tensor, tensor, ctx);
    } catch (...) {
        dlpack_legacy_export_deleter(managed);
        throw;
    }
    return make_dlpack_capsule(managed, "dltensor");
}

nb::tuple tensor_dlpack_device_tuple(Tensor& tensor) {
    if (tensor.buffer == nullptr || tensor.buffer->context == nullptr) {
        throw StatusError(NAVATALA_INVALID_HANDLE, "tensor is closed");
    }
    NavatalaBackend backend = tensor.buffer->context->backend;
    return nb::make_tuple(static_cast<int>(dlpack_device_type(backend)), tensor.buffer->context->device_id);
}

Tensor import_dlpack_tensor(DLTensor& dl_tensor, std::shared_ptr<void> owner, nb::object requested_device) {
    validate_compact_row_major(dl_tensor);
    if (dl_tensor.byte_offset > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        raise_python_runtime_error("DLPack byte_offset is too large");
    }
    NavatalaBackend backend = backend_from_dlpack_device(dl_tensor.device.device_type);
    if (backend != NAVATALA_BACKEND_CUDA_FFI && backend != NAVATALA_BACKEND_HIP_FFI) {
        raise_python_runtime_error("DLPack zero-copy import is currently implemented for CUDA/HIP tensors only; use Tensor.from_numpy() for explicit copies");
    }
    NavatalaBackend requested_backend = backend_from_device_object(requested_device);
    if (requested_backend != NAVATALA_BACKEND_AUTO_FFI && requested_backend != backend) {
        raise_python_runtime_error("Cannot zero-copy DLPack tensor into requested backend " + backend_name(requested_backend) + "; use explicit copy");
    }
    auto shape = shape_from_dlpack_tensor(dl_tensor);
    std::string dtype = dtype_from_dlpack(dl_tensor.dtype);
    std::size_t bytes = dlpack_tensor_nbytes(dl_tensor);
    auto* raw = static_cast<std::uint8_t*>(dl_tensor.data) + static_cast<std::ptrdiff_t>(dl_tensor.byte_offset);
    auto context = std::make_shared<ContextControlBlock>(backend, dl_tensor.device.device_id);
    auto buffer = std::make_shared<BufferControlBlock>(context, raw, bytes, std::move(owner));
    Tensor tensor(std::move(buffer), shape, dtype, backend_name(backend));
    return tensor;
}

Tensor tensor_from_dlpack_object(nb::object obj, nb::object device, nb::object queue) {
    nb::object capsule_obj = obj;
    if (!PyCapsule_CheckExact(capsule_obj.ptr())) {
        if (!PyObject_HasAttrString(obj.ptr(), "__dlpack__")) {
            raise_python_type_error("Tensor.from_dlpack expects a DLPack capsule or object implementing __dlpack__");
        }
        nb::object stream_token = dlpack_stream_from_queue_object(queue);
        if (stream_token.is_none()) {
            capsule_obj = obj.attr("__dlpack__")();
        } else {
            capsule_obj = obj.attr("__dlpack__")(stream_token);
        }
    }
    if (!PyCapsule_CheckExact(capsule_obj.ptr())) {
        raise_python_type_error("__dlpack__ did not return a PyCapsule");
    }
    if (PyCapsule_IsValid(capsule_obj.ptr(), "dltensor_versioned")) {
        auto* managed = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(capsule_obj.ptr(), "dltensor_versioned"));
        if (managed == nullptr) {
            throw nb::python_error();
        }
        if (managed->version.major != DLPACK_MAJOR_VERSION) {
            if (managed->deleter != nullptr) managed->deleter(managed);
            PyCapsule_SetName(capsule_obj.ptr(), "used_dltensor_versioned");
            raise_python_value_error("Unsupported DLPack major version");
        }
        PyCapsule_SetName(capsule_obj.ptr(), "used_dltensor_versioned");
        std::shared_ptr<void> owner(managed, [](void* ptr) noexcept {
            try {
                auto* self = static_cast<DLManagedTensorVersioned*>(ptr);
                if (self != nullptr && self->deleter != nullptr) self->deleter(self);
            } catch (...) {
            }
        });
        Tensor tensor = import_dlpack_tensor(managed->dl_tensor, std::move(owner), device);
        tensor.readonly = (managed->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0;
        return tensor;
    }
    if (PyCapsule_IsValid(capsule_obj.ptr(), "dltensor")) {
        auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule_obj.ptr(), "dltensor"));
        if (managed == nullptr) {
            throw nb::python_error();
        }
        PyCapsule_SetName(capsule_obj.ptr(), "used_dltensor");
        std::shared_ptr<void> owner(managed, [](void* ptr) noexcept {
            try {
                auto* self = static_cast<DLManagedTensor*>(ptr);
                if (self != nullptr && self->deleter != nullptr) self->deleter(self);
            } catch (...) {
            }
        });
        return import_dlpack_tensor(managed->dl_tensor, std::move(owner), device);
    }
    raise_python_value_error("Malformed or already-consumed DLPack capsule");
}

} // namespace navatala_gpu::python

NB_MODULE(navatala_gpu_ext, m) {
    m.doc() = "Generated navatala_gpu bindings";
    m.attr("__navatala_ffi_abi_version__") = NAVATALA_GPU_FFI_ABI_VERSION;
    m.attr("__navatala_python_binding_abi_version__") = 6;
    m.def("_get_abi_version", []() { return NAVATALA_GPU_FFI_ABI_VERSION; });
    m.def("get_capabilities", &navatala_gpu::python::get_capabilities_dict);
    nb::module_ m_runtime = m.def_submodule("runtime");
    m_runtime.def("queue_create", [](nb::object device, int priority) {
        return navatala_gpu::python::translate_status([&]() {
            return navatala_gpu::python::create_queue_capsule(device, priority);
        }, "queue_create");
    }, nb::arg("device") = nb::none(), nb::arg("priority") = -1);
    m_runtime.def("queue_sync", [](nb::object queue) {
        navatala_gpu::python::translate_status([&]() {
            navatala_gpu::python::sync_queue_object(queue);
            return 0;
        }, "queue_sync");
    }, nb::arg("queue"));
    m_runtime.def("queue_close", [](nb::object queue_handle) {
        navatala_gpu::python::translate_status([&]() {
            navatala_gpu::python::close_queue_capsule(queue_handle);
            return 0;
        }, "queue_close");
    }, nb::arg("queue_handle"));
    m_runtime.def("event_create", [](nb::object device) {
        return navatala_gpu::python::translate_status([&]() {
            return navatala_gpu::python::create_event_capsule(device);
        }, "event_create");
    }, nb::arg("device") = nb::none());
    m_runtime.def("event_create_from_queue", [](nb::object queue) {
        return navatala_gpu::python::translate_status([&]() {
            return navatala_gpu::python::create_event_capsule_from_queue(queue);
        }, "event_create_from_queue");
    }, nb::arg("queue"));
    m_runtime.def("event_record", [](nb::object event, nb::object queue) {
        navatala_gpu::python::translate_status([&]() {
            navatala_gpu::python::record_event_object(event, queue);
            return 0;
        }, "event_record");
    }, nb::arg("event"), nb::arg("queue"));
    m_runtime.def("event_wait", [](nb::object queue, nb::object event) {
        navatala_gpu::python::translate_status([&]() {
            navatala_gpu::python::wait_event_object(queue, event);
            return 0;
        }, "event_wait");
    }, nb::arg("queue"), nb::arg("event"));
    m_runtime.def("event_sync", [](nb::object event) {
        navatala_gpu::python::translate_status([&]() {
            navatala_gpu::python::sync_event_object(event);
            return 0;
        }, "event_sync");
    }, nb::arg("event"));
    m_runtime.def("event_is_complete", [](nb::object event) {
        return navatala_gpu::python::translate_status([&]() {
            return navatala_gpu::python::event_is_complete_object(event);
        }, "event_is_complete");
    }, nb::arg("event"));
    m_runtime.def("event_close", [](nb::object event_handle) {
        navatala_gpu::python::translate_status([&]() {
            navatala_gpu::python::close_event_capsule(event_handle);
            return 0;
        }, "event_close");
    }, nb::arg("event_handle"));
    nb::class_<navatala_gpu::python::Buffer>(m, "Buffer")
        .def_prop_ro("nbytes", &navatala_gpu::python::Buffer::nbytes)
        .def_prop_ro("device", &navatala_gpu::python::Buffer::device)
        .def_prop_ro("closed", &navatala_gpu::python::Buffer::is_closed)
        .def("close", [](navatala_gpu::python::Buffer& buffer) {
            return navatala_gpu::python::translate_status([&]() {
                buffer.close();
                return nb::none();
            }, "Buffer.close");
        });
    m_runtime.def("buffer_empty", [](std::uint64_t bytes, nb::object device, nb::object queue) {
        (void) queue;
        return navatala_gpu::python::translate_status([&]() {
            if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "buffer byte count overflows size_t");
            }
            return navatala_gpu::python::allocate_buffer(
                static_cast<std::size_t>(bytes),
                navatala_gpu::python::backend_from_device_object(device),
                navatala_gpu::python::device_id_from_device_object(device));
        }, "buffer_empty");
    }, nb::arg("bytes"), nb::arg("device") = nb::none(), nb::arg("queue") = nb::none());
    m_runtime.def("buffer_close", [](navatala_gpu::python::Buffer& buffer) {
        return navatala_gpu::python::translate_status([&]() {
            buffer.close();
            return nb::none();
        }, "buffer_close");
    }, nb::arg("buffer"));
    nb::class_<navatala_gpu::python::Tensor>(m, "Tensor")
        .def_prop_ro("shape", &navatala_gpu::python::Tensor::shape)
        .def_prop_ro("dtype", &navatala_gpu::python::Tensor::dtype)
        .def_prop_ro("device", &navatala_gpu::python::Tensor::device)
        .def_prop_ro("nbytes", &navatala_gpu::python::Tensor::nbytes)
        .def_prop_ro("readonly", [](const navatala_gpu::python::Tensor& tensor) { return tensor.readonly; })
        .def_prop_ro("closed", &navatala_gpu::python::Tensor::is_closed)
        .def("close", [](navatala_gpu::python::Tensor& tensor) {
            return navatala_gpu::python::translate_status([&]() {
                tensor.close();
                return nb::none();
            }, "Tensor.close");
        })
        .def("to_numpy", [](navatala_gpu::python::Tensor& tensor, nb::object queue) {
            return navatala_gpu::python::translate_status([&]() {
                return navatala_gpu::python::tensor_to_numpy_array(tensor, queue);
            }, "Tensor.to_numpy");
        }, nb::arg("queue") = nb::none())
        .def("to_dlpack", [](navatala_gpu::python::Tensor& tensor, nb::object stream, nb::object max_version) {
            return navatala_gpu::python::translate_status([&]() {
                return navatala_gpu::python::tensor_to_dlpack_capsule(tensor, stream, max_version);
            }, "Tensor.to_dlpack");
        }, nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none())
        .def("__dlpack__", [](navatala_gpu::python::Tensor& tensor, nb::object stream, nb::object max_version) {
            return navatala_gpu::python::translate_status([&]() {
                return navatala_gpu::python::tensor_to_dlpack_capsule(tensor, stream, max_version);
            }, "Tensor.__dlpack__");
        }, nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none())
        .def("__dlpack_device__", [](navatala_gpu::python::Tensor& tensor) {
            return navatala_gpu::python::translate_status([&]() {
                return navatala_gpu::python::tensor_dlpack_device_tuple(tensor);
            }, "Tensor.__dlpack_device__");
        });
    m_runtime.def("tensor_empty", [](std::vector<std::uint64_t> shape, std::string dtype, nb::object device, nb::object queue) {
        (void) queue;
        return navatala_gpu::python::translate_status([&]() {
            return navatala_gpu::python::allocate_tensor(
                shape, dtype,
                navatala_gpu::python::backend_from_device_object(device),
                navatala_gpu::python::device_id_from_device_object(device));
        }, "tensor_empty");
    }, nb::arg("shape"), nb::arg("dtype"), nb::arg("device") = nb::none(), nb::arg("queue") = nb::none());
    m_runtime.def("tensor_from_numpy", [](nb::object array, nb::object device, nb::object queue) {
        return navatala_gpu::python::translate_status([&]() {
            return navatala_gpu::python::tensor_from_numpy_like(array, device, queue);
        }, "tensor_from_numpy");
    }, nb::arg("array"), nb::arg("device") = nb::none(), nb::arg("queue") = nb::none());
    m_runtime.def("tensor_to_numpy", [](navatala_gpu::python::Tensor& tensor, nb::object queue) {
        return navatala_gpu::python::translate_status([&]() {
            return navatala_gpu::python::tensor_to_numpy_array(tensor, queue);
        }, "tensor_to_numpy");
    }, nb::arg("tensor"), nb::arg("queue") = nb::none());
    m_runtime.def("tensor_to_dlpack", [](navatala_gpu::python::Tensor& tensor, nb::object stream, nb::object max_version) {
        return navatala_gpu::python::translate_status([&]() {
            return navatala_gpu::python::tensor_to_dlpack_capsule(tensor, stream, max_version);
        }, "tensor_to_dlpack");
    }, nb::arg("tensor"), nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none());
    m_runtime.def("tensor_dlpack_device", [](navatala_gpu::python::Tensor& tensor) {
        return navatala_gpu::python::translate_status([&]() {
            return navatala_gpu::python::tensor_dlpack_device_tuple(tensor);
        }, "tensor_dlpack_device");
    }, nb::arg("tensor"));
    m_runtime.def("tensor_from_dlpack", [](nb::object obj, nb::object device, nb::object queue) {
        return navatala_gpu::python::translate_status([&]() {
            return navatala_gpu::python::tensor_from_dlpack_object(obj, device, queue);
        }, "tensor_from_dlpack");
    }, nb::arg("obj"), nb::arg("device") = nb::none(), nb::arg("queue") = nb::none());
    m_runtime.def("tensor_close", [](navatala_gpu::python::Tensor& tensor) {
        return navatala_gpu::python::translate_status([&]() {
            tensor.close();
            return nb::none();
        }, "tensor_close");
    }, nb::arg("tensor"));
    nb::module_ m_kernels = m.def_submodule("kernels");
    m_kernels.def("launch", [](nb::object source, std::string entry_point, nb::sequence args, std::string source_kind, nb::object queue, std::vector<std::uint32_t> grid, std::vector<std::uint32_t> block, bool sync) {
        return navatala_gpu::python::translate_status([&]() {
            navatala_gpu::python::launch_kernel_object(source, entry_point, args, source_kind, queue, grid, block, sync);
            return nb::none();
        }, "kernels.launch");
    }, nb::arg("source"), nb::arg("entry_point"), nb::arg("args"), nb::kw_only(), nb::arg("source_kind") = "opencl_c", nb::arg("queue"), nb::arg("grid") = std::vector<std::uint32_t>{1, 1, 1}, nb::arg("block") = std::vector<std::uint32_t>{1, 1, 1}, nb::arg("sync") = false);
    nb::module_ m_linalg = m.def_submodule("linalg");
    nb::module_ m_dataframe = m.def_submodule("dataframe");
    nb::module_ m_ml = m.def_submodule("ml");
    nb::module_ m_sparse = m.def_submodule("sparse");
    nb::module_ m_graph = m.def_submodule("graph");

    m_linalg.def("axpy", [](navatala_gpu::python::Tensor x, navatala_gpu::python::Tensor y, double alpha, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: x and y must have identical shape
        // target: operation the upstream generator.StdLib.Blas.axpyFloat32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (x.is_closed() || y.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "axpy tensor is closed");
            }
            if (x.dtype_value != "float32" || y.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "axpy requires float32 tensors");
            }
            if (x.shape_values != y.shape_values) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "axpy requires x and y to have identical shape");
            }
            if (x.device_value != y.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "axpy requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != y.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            std::uint64_t elements64 = navatala_gpu::python::checked_element_count(x.shape_values);
            if (elements64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "axpy element count overflows size_t");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_axpy_f32(x.buffer->handle, y.buffer->handle, static_cast<float>(alpha), static_cast<std::size_t>(elements64), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_axpy_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, x);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return y;
        }, "linalg.axpy");
    }, "Compute y = alpha * x + y on the selected GPU backend.\n\nThe Python binding accepts DLPack-compatible tensors and returns the updated y tensor.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32\n- hip: float32\n- opencl: float32\n- vulkan: float32\n- metal: float32\n\nSynchronization:\nThe binding returns after enqueue unless the caller requests synchronization.", nb::arg("x"), nb::arg("y"), nb::arg("alpha") = 1.000000, nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_linalg.def("axpy_dispatch", [](navatala_gpu::python::Tensor x, navatala_gpu::python::Tensor y, double alpha, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: x and y must have identical shape
        // precondition[invalid_dtype]: x and y must have identical dtype
        // target: dispatch_group the upstream generator.Python.Samples.axpyDispatchGroup
        const std::string dispatch_dtype = x.dtype_value;
        const std::string dispatch_backend = (!backend.empty() && backend != "auto" ? backend : x.device_value);
        if (dispatch_dtype == "float32") {
            return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
                if (x.is_closed() || y.is_closed()) {
                    throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "axpy tensor is closed");
                }
                if (x.dtype_value != "float32" || y.dtype_value != "float32") {
                    throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "axpy requires float32 tensors");
                }
                if (x.shape_values != y.shape_values) {
                    throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "axpy requires x and y to have identical shape");
                }
                if (x.device_value != y.device_value) {
                    throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "axpy requires tensors on the same backend");
                }
                if (!backend.empty() && backend != "auto" && backend != y.device_value) {
                    throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
                }
                std::uint64_t elements64 = navatala_gpu::python::checked_element_count(x.shape_values);
                if (elements64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                    throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "axpy element count overflows size_t");
                }
                if (sync && queue.is_none()) {
                    throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
                }
                navatala_gpu::python::check_status(
                    navatala_gpu_axpy_f32(x.buffer->handle, y.buffer->handle, static_cast<float>(alpha), static_cast<std::size_t>(elements64), navatala_gpu::python::queue_handle_from_object(queue)),
                    "navatala_gpu_axpy_f32");
                navatala_gpu::python::retain_tensor_until_queue_sync(queue, x);
                navatala_gpu::python::retain_tensor_until_queue_sync(queue, y);
                if (sync) {
                    navatala_gpu::python::sync_queue_object(queue);
                }
                return y;
            }, "linalg.axpy");
        }
        PyErr_SetString(PyExc_NotImplementedError, ("no dispatch variant is linked for dtype=" + dispatch_dtype + ", backend=" + dispatch_backend).c_str());
        throw nb::python_error();
    }, "Compute y = alpha * x + y through a dtype-dispatched binding group.\n\nThe Python binding validates dispatch-group dtype routing before invoking the concrete AXPY implementation.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32\n- hip: float32\n- opencl: float32\n- vulkan: float32\n- metal: float32\n\nSynchronization:\nThe binding returns after enqueue unless the caller requests synchronization.", nb::arg("x"), nb::arg("y"), nb::arg("alpha") = 1.000000, nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_linalg.def("gemm", [](navatala_gpu::python::Tensor a, navatala_gpu::python::Tensor b, navatala_gpu::python::Tensor c, double alpha, double beta, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: matrix dimensions must satisfy A[m,k] * B[k,n] -> C[m,n]
        // target: operation the upstream generator.StdLib.BlasExtended.gemmF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (a.is_closed() || b.is_closed() || c.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "gemm tensor is closed");
            }
            if (a.dtype_value != "float32" || b.dtype_value != "float32" || c.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gemm requires float32 tensors");
            }
            if (a.shape_values.size() != 2 || b.shape_values.size() != 2 || c.shape_values.size() != 2) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gemm requires rank-2 tensors");
            }
            const std::uint64_t m64 = a.shape_values[0];
            const std::uint64_t k64 = a.shape_values[1];
            const std::uint64_t b_k64 = b.shape_values[0];
            const std::uint64_t n64 = b.shape_values[1];
            if (b_k64 != k64 || c.shape_values[0] != m64 || c.shape_values[1] != n64) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gemm shape mismatch: A[m,k] * B[k,n] -> C[m,n]");
            }
            if (a.device_value != b.device_value || a.device_value != c.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gemm requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != c.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            const std::uint64_t maxSize = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
            if (m64 > maxSize || n64 > maxSize || k64 > maxSize) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "gemm dimension overflows size_t");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_gemm_f32(a.buffer->handle, b.buffer->handle, c.buffer->handle,
                    static_cast<std::size_t>(m64), static_cast<std::size_t>(n64), static_cast<std::size_t>(k64),
                    static_cast<float>(alpha), static_cast<float>(beta), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_gemm_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, a);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, b);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, c);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return c;
        }, "linalg.gemm");
    }, "Compute C = alpha * A @ B + beta * C on the selected GPU backend.\n\nThe Python binding validates rank-2 tensors and backend dtype support before dispatch.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32\n- hip: float32\n- opencl: float32\n- vulkan: float32\n- metal: float32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("a"), nb::arg("b"), nb::arg("c"), nb::arg("alpha") = 1.000000, nb::arg("beta") = 0.000000, nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_linalg.def("nrm2", [](navatala_gpu::python::Tensor x, navatala_gpu::python::Tensor result, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: nrm2 result must be a scalar float32 tensor
        // target: operation the upstream generator.StdLib.Blas.nrm2Float32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (x.is_closed() || result.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "nrm2 tensor is closed");
            }
            if (x.dtype_value != "float32" || result.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "nrm2 requires float32 tensors");
            }
            if (!result.shape_values.empty()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "nrm2 result must be a scalar tensor");
            }
            if (x.device_value != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "nrm2 requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            std::uint64_t elements64 = navatala_gpu::python::checked_element_count(x.shape_values);
            if (elements64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "nrm2 element count overflows size_t");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_nrm2_f32(x.buffer->handle, result.buffer->handle, static_cast<std::size_t>(elements64), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_nrm2_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, x);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, result);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return result;
        }, "linalg.nrm2");
    }, "Compute result[0] = ||x||_2 on the selected GPU backend.\n\nThe result argument is an explicit scalar float32 Tensor, keeping output allocation visible to the caller.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32\n- hip: float32\n- opencl: float32\n- vulkan: float32\n- metal: float32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("x"), nb::arg("result"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("reduce_sum", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data length must be count and input_valid length must be ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Reduce.reduceSumF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "reduce_sum tensor is closed");
            }
            if (data.dtype_value != "float32" || output.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_sum requires float32 data and output tensors");
            }
            if (input_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_sum requires uint32 validity tensors");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || !output.shape_values.empty() || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_sum expects rank-1 data/validity, scalar output, and rank-1 output_valid");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "reduce_sum count exceeds uint32 index range");
            }
            const std::uint64_t expected_valid_words = (count + 31U) / 32U;
            if (data.shape_values[0] != count || input_valid.shape_values[0] != expected_valid_words || output_valid.shape_values[0] != 1U) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_sum shape mismatch for count and validity bitmap");
            }
            if (data.device_value != input_valid.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_sum requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_reduce_sum_f32(data.buffer->handle, input_valid.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_reduce_sum_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.reduce_sum");
    }, "Compute a null-aware float32 column sum.\n\nThe input_valid bitmap uses uint32 words with one bit per row. output is a scalar float32 tensor; output_valid bit 0 is set when at least one element is valid.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32, uint32\n- hip: float32, uint32\n- opencl: float32, uint32\n- vulkan: float32, uint32\n- metal: float32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("output"), nb::arg("output_valid"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("reduce_sum_i32_to_i64", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data length must be count and input_valid length must be ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Reduce.reduceSumI32ToI64
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "reduce_sum_i32_to_i64 tensor is closed");
            }
            if (data.dtype_value != "int32" || output.dtype_value != "int64") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_sum_i32_to_i64 requires int32 data and int64 output tensors");
            }
            if (input_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_sum_i32_to_i64 requires uint32 validity tensors");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || !output.shape_values.empty() || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_sum_i32_to_i64 expects rank-1 data/validity, scalar output, and rank-1 output_valid");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "reduce_sum_i32_to_i64 count exceeds uint32 index range");
            }
            const std::uint64_t expected_valid_words = (count + 31U) / 32U;
            if (data.shape_values[0] != count || input_valid.shape_values[0] != expected_valid_words || output_valid.shape_values[0] != 1U) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_sum_i32_to_i64 shape mismatch for count and validity bitmap");
            }
            if (data.device_value != input_valid.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_sum_i32_to_i64 requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_reduce_sum_i32_to_i64(data.buffer->handle, input_valid.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_reduce_sum_i32_to_i64");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.reduce_sum_i32_to_i64");
    }, "Compute a null-aware int32 column sum widened to int64.\n\nThe input_valid bitmap uses uint32 words with one bit per row. output is a scalar int64 tensor; output_valid bit 0 is set when at least one element is valid. Nulls are skipped and values are widened before accumulation.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: int32, int64, uint32\n- hip: int32, int64, uint32\n- opencl: int32, int64, uint32\n- vulkan: int32, int64, uint32\n- metal: int32, int64, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("output"), nb::arg("output_valid"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("reduce_min_i32", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data length must be count and input_valid length must be ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Reduce.reduceMinI32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "reduce_min_i32 tensor is closed");
            }
            if (data.dtype_value != "int32" || output.dtype_value != "int32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_min_i32 requires int32 data and output tensors");
            }
            if (input_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_min_i32 requires uint32 validity tensors");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || !output.shape_values.empty() || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_min_i32 expects rank-1 data/validity, scalar output, and rank-1 output_valid");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "reduce_min_i32 count exceeds uint32 index range");
            }
            const std::uint64_t expected_valid_words = (count + 31U) / 32U;
            if (data.shape_values[0] != count || input_valid.shape_values[0] != expected_valid_words || output_valid.shape_values[0] != 1U) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_min_i32 shape mismatch for count and validity bitmap");
            }
            if (data.device_value != input_valid.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_min_i32 requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_reduce_min_i32(data.buffer->handle, input_valid.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_reduce_min_i32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.reduce_min_i32");
    }, "Compute a null-aware int32 column minimum.\n\nThe input_valid bitmap uses uint32 words with one bit per row. output is a scalar int32 tensor; output_valid bit 0 is set when at least one element is valid.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: int32, uint32\n- hip: int32, uint32\n- opencl: int32, uint32\n- vulkan: int32, uint32\n- metal: int32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("output"), nb::arg("output_valid"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("reduce_max_i32", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data length must be count and input_valid length must be ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Reduce.reduceMaxI32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "reduce_max_i32 tensor is closed");
            }
            if (data.dtype_value != "int32" || output.dtype_value != "int32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_max_i32 requires int32 data and output tensors");
            }
            if (input_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_max_i32 requires uint32 validity tensors");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || !output.shape_values.empty() || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_max_i32 expects rank-1 data/validity, scalar output, and rank-1 output_valid");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "reduce_max_i32 count exceeds uint32 index range");
            }
            const std::uint64_t expected_valid_words = (count + 31U) / 32U;
            if (data.shape_values[0] != count || input_valid.shape_values[0] != expected_valid_words || output_valid.shape_values[0] != 1U) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_max_i32 shape mismatch for count and validity bitmap");
            }
            if (data.device_value != input_valid.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "reduce_max_i32 requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_reduce_max_i32(data.buffer->handle, input_valid.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_reduce_max_i32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.reduce_max_i32");
    }, "Compute a null-aware int32 column maximum.\n\nThe input_valid bitmap uses uint32 words with one bit per row. output is a scalar int32 tensor; output_valid bit 0 is set when at least one element is valid.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: int32, uint32\n- hip: int32, uint32\n- opencl: int32, uint32\n- vulkan: int32, uint32\n- metal: int32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("output"), nb::arg("output_valid"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("fill_null", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, double fill_value, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data/output length must be count and validity bitmap length must be ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.NullOps.fillNullF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "fill_null tensor is closed");
            }
            if (data.dtype_value != "float32" || output.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_null requires float32 data and output tensors");
            }
            if (input_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_null requires uint32 validity tensors");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || output.shape_values.size() != 1 || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_null expects rank-1 data/output and validity tensors");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "fill_null count exceeds uint32 index range");
            }
            const std::uint64_t valid_words = (count + 31U) / 32U;
            if (data.shape_values[0] != count || output.shape_values[0] != count || input_valid.shape_values[0] != valid_words || output_valid.shape_values[0] != valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_null shape mismatch for count and validity bitmap");
            }
            if (data.device_value != input_valid.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_null requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_fill_null_f32(data.buffer->handle, input_valid.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<float>(fill_value), static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_fill_null_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.fill_null");
    }, "Replace null float32 column entries with a scalar fill value.\n\nThe input_valid bitmap uses uint32 words with one bit per row. output_valid is overwritten so every row in [0, count) is valid.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32, uint32\n- hip: float32, uint32\n- opencl: float32, uint32\n- vulkan: float32, uint32\n- metal: float32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("output"), nb::arg("output_valid"), nb::arg("fill_value"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("fill_null_i32", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::int64_t fill_value, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data/output length must be count, fill_value must fit in int32, and validity bitmap length must be ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.NullOps.fillNullI32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "fill_null_i32 tensor is closed");
            }
            if (data.dtype_value != "int32" || output.dtype_value != "int32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_null_i32 requires int32 data and output tensors");
            }
            if (input_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_null_i32 requires uint32 validity tensors");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || output.shape_values.size() != 1 || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_null_i32 expects rank-1 data, output, and validity tensors");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "fill_null_i32 count exceeds uint32 index range");
            }
            const std::uint64_t expected_valid_words = (count + 31U) / 32U;
            if (data.shape_values[0] != count || output.shape_values[0] != count || input_valid.shape_values[0] != expected_valid_words || output_valid.shape_values[0] != expected_valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_null_i32 shape mismatch for count and validity bitmaps");
            }
            if (fill_value < static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()) || fill_value > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "fill_null_i32 fill_value is outside int32 range");
            }
            if (data.device_value != input_valid.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_null_i32 requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_fill_null_i32(data.buffer->handle, input_valid.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::int32_t>(fill_value), static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_fill_null_i32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.fill_null_i32");
    }, "Replace null int32 column entries with a scalar fill value.\n\nThe input_valid bitmap uses uint32 words with one bit per row. output_valid is overwritten so every row in [0, count) is valid.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: int32, uint32\n- hip: int32, uint32\n- opencl: int32, uint32\n- vulkan: int32, uint32\n- metal: int32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("output"), nb::arg("output_valid"), nb::arg("fill_value"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("gather", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor indices, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t input_count, std::uint64_t output_count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data length must be input_count; indices/output length must be output_count; validity lengths must match ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Scatter.gatherF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || indices.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "gather tensor is closed");
            }
            if (data.dtype_value != "float32" || output.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gather requires float32 data and output tensors");
            }
            if (input_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gather requires uint32 validity tensors");
            }
            if (indices.dtype_value != "int32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gather requires int32 indices");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || indices.shape_values.size() != 1 || output.shape_values.size() != 1 || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gather expects rank-1 data, indices, output, and validity tensors");
            }
            if (input_count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) || output_count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "gather counts exceed uint32 index range");
            }
            const std::uint64_t input_valid_words = (input_count + 31U) / 32U;
            const std::uint64_t output_valid_words = (output_count + 31U) / 32U;
            if (data.shape_values[0] != input_count || input_valid.shape_values[0] != input_valid_words || indices.shape_values[0] != output_count || output.shape_values[0] != output_count || output_valid.shape_values[0] != output_valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gather shape mismatch for counts and validity bitmaps");
            }
            if (data.device_value != input_valid.device_value || data.device_value != indices.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gather requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_gather_f32(data.buffer->handle, input_valid.buffer->handle, indices.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(input_count), static_cast<std::size_t>(output_count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_gather_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, indices);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.gather");
    }, "Gather float32 DataFrame values by int32 row indices.\n\nInvalid or out-of-range source rows become null output rows. output_valid is overwritten with ceil(output_count / 32) uint32 words.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32, uint32, int32\n- hip: float32, uint32, int32\n- opencl: float32, uint32, int32\n- vulkan: float32, uint32, int32\n- metal: float32, uint32, int32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("indices"), nb::arg("output"), nb::arg("output_valid"), nb::arg("input_count"), nb::arg("output_count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("gather_i32", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor indices, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t input_count, std::uint64_t output_count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data length/input validity bitmap/indices/output/output validity must match input_count and output_count
        // target: operation the upstream generator.StdLib.DataFrame.Scatter.gatherI32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || indices.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "gather_i32 tensor is closed");
            }
            if (data.dtype_value != "int32" || output.dtype_value != "int32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gather_i32 requires int32 data and output tensors");
            }
            if (input_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gather_i32 requires uint32 validity tensors");
            }
            if (indices.dtype_value != "int32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gather_i32 requires int32 indices");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || indices.shape_values.size() != 1 || output.shape_values.size() != 1 || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gather_i32 expects rank-1 data, indices, output, and validity tensors");
            }
            if (input_count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) || output_count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "gather_i32 counts exceed uint32 index range");
            }
            const std::uint64_t input_valid_words = (input_count + 31U) / 32U;
            const std::uint64_t output_valid_words = (output_count + 31U) / 32U;
            if (data.shape_values[0] != input_count || input_valid.shape_values[0] != input_valid_words || indices.shape_values[0] != output_count || output.shape_values[0] != output_count || output_valid.shape_values[0] != output_valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gather_i32 shape mismatch for counts and validity bitmaps");
            }
            if (data.device_value != input_valid.device_value || data.device_value != indices.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "gather_i32 requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_gather_i32(data.buffer->handle, input_valid.buffer->handle, indices.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(input_count), static_cast<std::size_t>(output_count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_gather_i32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, indices);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.gather_i32");
    }, "Gather int32 DataFrame values by int32 row indices.\n\nOut-of-range indices and null source rows produce output value 0 and clear the corresponding output validity bit.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: int32, uint32\n- hip: int32, uint32\n- opencl: int32, uint32\n- vulkan: int32, uint32\n- metal: int32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("indices"), nb::arg("output"), nb::arg("output_valid"), nb::arg("input_count"), nb::arg("output_count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("scatter", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor indices, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t input_count, std::uint64_t output_count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data/indices length must be input_count; output length must be output_count; validity lengths must match ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Scatter.scatterF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || indices.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "scatter tensor is closed");
            }
            if (data.dtype_value != "float32" || output.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "scatter requires float32 data and output tensors");
            }
            if (input_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "scatter requires uint32 validity tensors");
            }
            if (indices.dtype_value != "int32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "scatter requires int32 indices");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || indices.shape_values.size() != 1 || output.shape_values.size() != 1 || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "scatter expects rank-1 data, indices, output, and validity tensors");
            }
            if (input_count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) || output_count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "scatter counts exceed uint32 index range");
            }
            const std::uint64_t input_valid_words = (input_count + 31U) / 32U;
            const std::uint64_t output_valid_words = (output_count + 31U) / 32U;
            if (data.shape_values[0] != input_count || input_valid.shape_values[0] != input_valid_words || indices.shape_values[0] != input_count || output.shape_values[0] != output_count || output_valid.shape_values[0] != output_valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "scatter shape mismatch for counts and validity bitmaps");
            }
            if (data.device_value != input_valid.device_value || data.device_value != indices.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "scatter requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_scatter_f32(data.buffer->handle, input_valid.buffer->handle, indices.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(input_count), static_cast<std::size_t>(output_count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_scatter_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, indices);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.scatter");
    }, "Scatter float32 DataFrame values by int32 destination indices.\n\nFor every in-range index, output[indices[i]] receives data[i]. Valid source rows set the destination validity bit; invalid source rows leave validity unchanged. Duplicate destination indices have contract-level undefined ordering.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32, uint32, int32\n- hip: float32, uint32, int32\n- opencl: float32, uint32, int32\n- vulkan: float32, uint32, int32\n- metal: float32, uint32, int32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("indices"), nb::arg("output"), nb::arg("output_valid"), nb::arg("input_count"), nb::arg("output_count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("scatter_i32", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor indices, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t input_count, std::uint64_t output_count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data/indices length must be input_count; output length must be output_count; validity lengths must match ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Scatter.scatterI32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || indices.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "scatter_i32 tensor is closed");
            }
            if (data.dtype_value != "int32" || output.dtype_value != "int32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "scatter_i32 requires int32 data and output tensors");
            }
            if (input_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "scatter_i32 requires uint32 validity tensors");
            }
            if (indices.dtype_value != "int32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "scatter_i32 requires int32 indices");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || indices.shape_values.size() != 1 || output.shape_values.size() != 1 || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "scatter_i32 expects rank-1 data, indices, output, and validity tensors");
            }
            if (input_count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) || output_count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "scatter_i32 counts exceed uint32 index range");
            }
            const std::uint64_t input_valid_words = (input_count + 31U) / 32U;
            const std::uint64_t output_valid_words = (output_count + 31U) / 32U;
            if (data.shape_values[0] != input_count || input_valid.shape_values[0] != input_valid_words || indices.shape_values[0] != input_count || output.shape_values[0] != output_count || output_valid.shape_values[0] != output_valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "scatter_i32 shape mismatch for counts and validity bitmaps");
            }
            if (data.device_value != input_valid.device_value || data.device_value != indices.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "scatter_i32 requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_scatter_i32(data.buffer->handle, input_valid.buffer->handle, indices.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(input_count), static_cast<std::size_t>(output_count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_scatter_i32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, indices);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.scatter_i32");
    }, "Scatter int32 DataFrame values by int32 destination indices.\n\nFor every in-range index, output[indices[i]] receives data[i]. Valid source rows set the destination validity bit; invalid source rows leave validity unchanged. Duplicate destination indices have contract-level undefined ordering.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: int32, uint32\n- hip: int32, uint32\n- opencl: int32, uint32\n- vulkan: int32, uint32\n- metal: int32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("indices"), nb::arg("output"), nb::arg("output_valid"), nb::arg("input_count"), nb::arg("output_count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("compact_by_mask", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor mask, navatala_gpu::python::Tensor prefix_sum, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data/mask/prefix_sum length must be count; input validity length must be ceil(count / 32); output validity length must be ceil(output length / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Compact.compactByMaskF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || mask.is_closed() || prefix_sum.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "compact_by_mask tensor is closed");
            }
            if (data.dtype_value != "float32" || output.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "compact_by_mask requires float32 data and output tensors");
            }
            if (input_valid.dtype_value != "uint32" || mask.dtype_value != "uint32" || prefix_sum.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "compact_by_mask requires uint32 validity, mask, and prefix-sum tensors");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || mask.shape_values.size() != 1 || prefix_sum.shape_values.size() != 1 || output.shape_values.size() != 1 || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "compact_by_mask expects rank-1 data, mask, prefix_sum, output, and validity tensors");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) || output.shape_values[0] > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "compact_by_mask counts exceed uint32 index range");
            }
            const std::uint64_t input_valid_words = (count + 31U) / 32U;
            const std::uint64_t output_count = output.shape_values[0];
            const std::uint64_t output_valid_words = (output_count + 31U) / 32U;
            if (data.shape_values[0] != count || input_valid.shape_values[0] != input_valid_words || mask.shape_values[0] != count || prefix_sum.shape_values[0] != count || output_valid.shape_values[0] != output_valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "compact_by_mask shape mismatch for count and validity bitmaps");
            }
            if (data.device_value != input_valid.device_value || data.device_value != mask.device_value || data.device_value != prefix_sum.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "compact_by_mask requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_compact_by_mask_f32(data.buffer->handle, input_valid.buffer->handle, mask.buffer->handle, prefix_sum.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(count), static_cast<std::size_t>(output_count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_compact_by_mask_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, mask);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, prefix_sum);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.compact_by_mask");
    }, "Compact float32 DataFrame values by a uint32 selection mask.\n\nRows with mask[i] == 1 are copied to output[prefix_sum[i]]. Source validity bits are propagated to output_valid, which is overwritten.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32, uint32\n- hip: float32, uint32\n- opencl: float32, uint32\n- vulkan: float32, uint32\n- metal: float32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("mask"), nb::arg("prefix_sum"), nb::arg("output"), nb::arg("output_valid"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("compact_by_mask_i32", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor mask, navatala_gpu::python::Tensor prefix_sum, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data/mask/prefix_sum length must be count; input validity length must be ceil(count / 32); output validity length must be ceil(output length / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Compact.compactByMaskI32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || mask.is_closed() || prefix_sum.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "compact_by_mask_i32 tensor is closed");
            }
            if (data.dtype_value != "int32" || output.dtype_value != "int32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "compact_by_mask_i32 requires int32 data and output tensors");
            }
            if (input_valid.dtype_value != "uint32" || mask.dtype_value != "uint32" || prefix_sum.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "compact_by_mask_i32 requires uint32 validity, mask, and prefix-sum tensors");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || mask.shape_values.size() != 1 || prefix_sum.shape_values.size() != 1 || output.shape_values.size() != 1 || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "compact_by_mask_i32 expects rank-1 data, mask, prefix_sum, output, and validity tensors");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) || output.shape_values[0] > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "compact_by_mask_i32 counts exceed uint32 index range");
            }
            const std::uint64_t input_valid_words = (count + 31U) / 32U;
            const std::uint64_t output_count = output.shape_values[0];
            const std::uint64_t output_valid_words = (output_count + 31U) / 32U;
            if (data.shape_values[0] != count || input_valid.shape_values[0] != input_valid_words || mask.shape_values[0] != count || prefix_sum.shape_values[0] != count || output_valid.shape_values[0] != output_valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "compact_by_mask_i32 shape mismatch for count and validity bitmaps");
            }
            if (data.device_value != input_valid.device_value || data.device_value != mask.device_value || data.device_value != prefix_sum.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "compact_by_mask_i32 requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_compact_by_mask_i32(data.buffer->handle, input_valid.buffer->handle, mask.buffer->handle, prefix_sum.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(count), static_cast<std::size_t>(output_count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_compact_by_mask_i32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, mask);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, prefix_sum);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.compact_by_mask_i32");
    }, "Compact int32 DataFrame values by a uint32 selection mask.\n\nRows with mask[i] == 1 are copied to output[prefix_sum[i]]. Source validity bits are propagated to output_valid, which is overwritten.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: int32, uint32\n- hip: int32, uint32\n- opencl: int32, uint32\n- vulkan: int32, uint32\n- metal: int32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("mask"), nb::arg("prefix_sum"), nb::arg("output"), nb::arg("output_valid"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("coalesce", [](navatala_gpu::python::Tensor col1, navatala_gpu::python::Tensor col1_valid, navatala_gpu::python::Tensor col2, navatala_gpu::python::Tensor col2_valid, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: input/output column lengths must be count and validity bitmap lengths must be ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.NullOps.coalesceF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (col1.is_closed() || col1_valid.is_closed() || col2.is_closed() || col2_valid.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "coalesce tensor is closed");
            }
            if (col1.dtype_value != "float32" || col2.dtype_value != "float32" || output.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "coalesce requires float32 input and output tensors");
            }
            if (col1_valid.dtype_value != "uint32" || col2_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "coalesce requires uint32 validity tensors");
            }
            if (col1.shape_values.size() != 1 || col2.shape_values.size() != 1 || output.shape_values.size() != 1 || col1_valid.shape_values.size() != 1 || col2_valid.shape_values.size() != 1 || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "coalesce expects rank-1 columns and validity tensors");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "coalesce count exceeds uint32 index range");
            }
            const std::uint64_t valid_words = (count + 31U) / 32U;
            if (col1.shape_values[0] != count || col2.shape_values[0] != count || output.shape_values[0] != count || col1_valid.shape_values[0] != valid_words || col2_valid.shape_values[0] != valid_words || output_valid.shape_values[0] != valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "coalesce shape mismatch for count and validity bitmaps");
            }
            if (col1.device_value != col1_valid.device_value || col1.device_value != col2.device_value || col1.device_value != col2_valid.device_value || col1.device_value != output.device_value || col1.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "coalesce requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_coalesce_f32(col1.buffer->handle, col1_valid.buffer->handle, col2.buffer->handle, col2_valid.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_coalesce_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, col1);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, col1_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, col2);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, col2_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.coalesce");
    }, "Select the first non-null value from two float32 DataFrame columns.\n\nRows prefer col1 when col1_valid is set; otherwise they use col2 when col2_valid is set. output_valid is overwritten with ceil(count / 32) uint32 words.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32, uint32\n- hip: float32, uint32\n- opencl: float32, uint32\n- vulkan: float32, uint32\n- metal: float32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("col1"), nb::arg("col1_valid"), nb::arg("col2"), nb::arg("col2_valid"), nb::arg("output"), nb::arg("output_valid"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("copy_if_valid", [](navatala_gpu::python::Tensor src, navatala_gpu::python::Tensor src_valid, navatala_gpu::python::Tensor dst_valid, navatala_gpu::python::Tensor dst, navatala_gpu::python::Tensor out_valid, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: src/dst lengths must be count and validity bitmap lengths must be ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.NullOps.copyIfValidF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (src.is_closed() || src_valid.is_closed() || dst_valid.is_closed() || dst.is_closed() || out_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "copy_if_valid tensor is closed");
            }
            if (src.dtype_value != "float32" || dst.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "copy_if_valid requires float32 source and destination tensors");
            }
            if (src_valid.dtype_value != "uint32" || dst_valid.dtype_value != "uint32" || out_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "copy_if_valid requires uint32 validity tensors");
            }
            if (src.shape_values.size() != 1 || dst.shape_values.size() != 1 || src_valid.shape_values.size() != 1 || dst_valid.shape_values.size() != 1 || out_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "copy_if_valid expects rank-1 columns and validity tensors");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "copy_if_valid count exceeds uint32 index range");
            }
            const std::uint64_t valid_words = (count + 31U) / 32U;
            if (src.shape_values[0] != count || dst.shape_values[0] != count || src_valid.shape_values[0] != valid_words || dst_valid.shape_values[0] != valid_words || out_valid.shape_values[0] != valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "copy_if_valid shape mismatch for count and validity bitmaps");
            }
            if (src.device_value != src_valid.device_value || src.device_value != dst_valid.device_value || src.device_value != dst.device_value || src.device_value != out_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "copy_if_valid requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != dst.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_copy_if_valid_f32(src.buffer->handle, src_valid.buffer->handle, dst_valid.buffer->handle, dst.buffer->handle, out_valid.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_copy_if_valid_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, src);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, src_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, dst_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, dst);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, out_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return dst;
        }, "dataframe.copy_if_valid");
    }, "Copy float32 values where source and destination rows are both valid.\n\ndst is updated in place only at the intersection of src_valid and dst_valid; other dst rows are unchanged. out_valid is overwritten with that intersection bitmap.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32, uint32\n- hip: float32, uint32\n- opencl: float32, uint32\n- vulkan: float32, uint32\n- metal: float32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("src"), nb::arg("src_valid"), nb::arg("dst_valid"), nb::arg("dst"), nb::arg("out_valid"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("inclusive_scan_sum_valid_prefix", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data/output length must be count <= 256 and validity bitmap length must be ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Scan.inclusiveScanSumF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "inclusive_scan_sum_valid_prefix tensor is closed");
            }
            if (data.dtype_value != "float32" || output.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "inclusive_scan_sum_valid_prefix requires float32 data and output tensors");
            }
            if (input_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "inclusive_scan_sum_valid_prefix requires uint32 validity tensors");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || output.shape_values.size() != 1 || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "inclusive_scan_sum_valid_prefix expects rank-1 data/output and validity tensors");
            }
            if (count > 256U) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "inclusive_scan_sum_valid_prefix count exceeds formal single-workgroup limit of 256");
            }
            const std::uint64_t valid_words = (count + 31U) / 32U;
            if (data.shape_values[0] != count || output.shape_values[0] != count || input_valid.shape_values[0] != valid_words || output_valid.shape_values[0] != valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "inclusive_scan_sum_valid_prefix shape mismatch for count and validity bitmaps");
            }
            if (data.device_value != input_valid.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "inclusive_scan_sum_valid_prefix requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_inclusive_scan_sum_f32(data.buffer->handle, input_valid.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_inclusive_scan_sum_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.inclusive_scan_sum_valid_prefix");
    }, "Compute a nullable float32 inclusive prefix sum with prefix-valid null propagation for up to 256 rows.\n\nNull rows contribute zero to the running sum. output_valid is set only for prefixes where every input row up to that point is valid.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32, uint32\n- hip: float32, uint32\n- opencl: float32, uint32\n- vulkan: float32, uint32\n- metal: float32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("output"), nb::arg("output_valid"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("inclusive_scan_sum_i32_to_i64_skip_nulls", [](navatala_gpu::python::Tensor data, navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor output, navatala_gpu::python::Tensor output_valid, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: data/output length must be count <= 256 and validity bitmap length must be ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Scan.inclusiveScanSumI32ToI64SkipNulls
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (data.is_closed() || input_valid.is_closed() || output.is_closed() || output_valid.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "inclusive_scan_sum_i32_to_i64_skip_nulls tensor is closed");
            }
            if (data.dtype_value != "int32" || output.dtype_value != "int64") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "inclusive_scan_sum_i32_to_i64_skip_nulls requires int32 data and int64 output tensors");
            }
            if (input_valid.dtype_value != "uint32" || output_valid.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "inclusive_scan_sum_i32_to_i64_skip_nulls requires uint32 validity tensors");
            }
            if (data.shape_values.size() != 1 || input_valid.shape_values.size() != 1 || output.shape_values.size() != 1 || output_valid.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "inclusive_scan_sum_i32_to_i64_skip_nulls expects rank-1 data/output and validity tensors");
            }
            if (count > 256U) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "inclusive_scan_sum_i32_to_i64_skip_nulls count exceeds formal single-workgroup limit of 256");
            }
            const std::uint64_t valid_words = (count + 31U) / 32U;
            if (data.shape_values[0] != count || output.shape_values[0] != count || input_valid.shape_values[0] != valid_words || output_valid.shape_values[0] != valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "inclusive_scan_sum_i32_to_i64_skip_nulls shape mismatch for count and validity bitmaps");
            }
            if (data.device_value != input_valid.device_value || data.device_value != output.device_value || data.device_value != output_valid.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "inclusive_scan_sum_i32_to_i64_skip_nulls requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_inclusive_scan_sum_i32_to_i64_skip_nulls(data.buffer->handle, input_valid.buffer->handle, output.buffer->handle, output_valid.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_inclusive_scan_sum_i32_to_i64_skip_nulls");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, data);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output_valid);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.inclusive_scan_sum_i32_to_i64_skip_nulls");
    }, "Compute a skip-null int32 inclusive prefix sum widened to int64 for up to 256 rows.\n\nNull rows contribute zero to the running sum and output_valid follows input_valid. output uses int64 values to avoid int32 cumulative-sum overflow.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: int32, int64, uint32\n- hip: int32, int64, uint32\n- opencl: int32, int64, uint32\n- vulkan: int32, int64, uint32\n- metal: int32, int64, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("data"), nb::arg("input_valid"), nb::arg("output"), nb::arg("output_valid"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("count_valid", [](navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor output, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: input_valid length must be ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Reduce.countValid
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (input_valid.is_closed() || output.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "count_valid tensor is closed");
            }
            if (input_valid.dtype_value != "uint32" || output.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "count_valid requires uint32 validity and output tensors");
            }
            if (input_valid.shape_values.size() != 1 || !output.shape_values.empty()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "count_valid expects rank-1 validity and scalar output tensors");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "count_valid count exceeds uint32 index range");
            }
            const std::uint64_t valid_words = (count + 31U) / 32U;
            if (input_valid.shape_values[0] != valid_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "count_valid shape mismatch for count and validity bitmap");
            }
            if (input_valid.device_value != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "count_valid requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_count_valid_u32(input_valid.buffer->handle, output.buffer->handle, static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_count_valid_u32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.count_valid");
    }, "Count valid rows in a uint32 validity bitmap.\n\nThe input_valid bitmap uses uint32 words with one bit per row. output is a scalar uint32 tensor.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: uint32\n- hip: uint32\n- opencl: uint32\n- vulkan: uint32\n- metal: uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("input_valid"), nb::arg("output"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("is_valid", [](navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor output, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: input_valid length must be ceil(count / 32) and output length must equal count
        // target: operation the upstream generator.StdLib.DataFrame.NullOps.isValid
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (input_valid.is_closed() || output.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "is_valid tensor is closed");
            }
            if (input_valid.dtype_value != "uint32" || output.dtype_value != "uint8") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "is_valid requires uint32 validity and uint8 output tensors");
            }
            if (input_valid.shape_values.size() != 1 || output.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "is_valid expects rank-1 validity and output tensors");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "is_valid count exceeds uint32 index range");
            }
            const std::uint64_t validity_words = (count + 31U) / 32U;
            if (input_valid.shape_values[0] != validity_words || output.shape_values[0] != count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "is_valid shape mismatch for count and validity bitmap");
            }
            if (input_valid.device_value != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "is_valid requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_is_valid_u8(input_valid.buffer->handle, output.buffer->handle, static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_is_valid_u8");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.is_valid");
    }, "Expand a validity bitmap into a uint8 valid mask.\n\ninput_valid is a rank-1 uint32 bitmap with ceil(count / 32) words. output is a rank-1 uint8 tensor of length count where 1 means valid.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: uint32, uint8\n- hip: uint32, uint8\n- opencl: uint32, uint8\n- vulkan: uint32, uint8\n- metal: uint32, uint8\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("input_valid"), nb::arg("output"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("is_null", [](navatala_gpu::python::Tensor input_valid, navatala_gpu::python::Tensor output, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: input_valid length must be ceil(count / 32) and output length must equal count
        // target: operation the upstream generator.StdLib.DataFrame.NullOps.isNull
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (input_valid.is_closed() || output.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "is_null tensor is closed");
            }
            if (input_valid.dtype_value != "uint32" || output.dtype_value != "uint8") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "is_null requires uint32 validity and uint8 output tensors");
            }
            if (input_valid.shape_values.size() != 1 || output.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "is_null expects rank-1 validity and output tensors");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "is_null count exceeds uint32 index range");
            }
            const std::uint64_t validity_words = (count + 31U) / 32U;
            if (input_valid.shape_values[0] != validity_words || output.shape_values[0] != count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "is_null shape mismatch for count and validity bitmap");
            }
            if (input_valid.device_value != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "is_null requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_is_null_u8(input_valid.buffer->handle, output.buffer->handle, static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_is_null_u8");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, input_valid);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.is_null");
    }, "Expand a validity bitmap into a uint8 null mask.\n\ninput_valid is a rank-1 uint32 bitmap with ceil(count / 32) words. output is a rank-1 uint8 tensor of length count where 1 means null.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: uint32, uint8\n- hip: uint32, uint8\n- opencl: uint32, uint8\n- vulkan: uint32, uint8\n- metal: uint32, uint8\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("input_valid"), nb::arg("output"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("fill_constant", [](navatala_gpu::python::Tensor output, double value, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: output length must equal count
        // target: operation the upstream generator.StdLib.DataFrame.Fill.fillConstantF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (output.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "fill_constant tensor is closed");
            }
            if (output.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_constant requires a float32 output tensor");
            }
            if (output.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_constant expects a rank-1 output tensor");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "fill_constant count exceeds uint32 index range");
            }
            if (output.shape_values[0] != count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_constant output length must equal count");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_fill_constant_f32(output.buffer->handle, static_cast<float>(value), static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_fill_constant_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.fill_constant");
    }, "Fill a float32 output tensor with a scalar value.\n\noutput is a rank-1 float32 tensor with length count. value is converted to float32 by the runtime ABI.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32\n- hip: float32\n- opencl: float32\n- vulkan: float32\n- metal: float32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("output"), nb::arg("value"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("fill_constant_i32", [](navatala_gpu::python::Tensor output, std::int64_t value, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: output length must equal count and value must fit in int32
        // target: operation the upstream generator.StdLib.DataFrame.Fill.fillConstantI32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (output.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "fill_constant_i32 tensor is closed");
            }
            if (output.dtype_value != "int32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_constant_i32 requires an int32 output tensor");
            }
            if (output.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_constant_i32 expects a rank-1 output tensor");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "fill_constant_i32 count exceeds uint32 index range");
            }
            if (output.shape_values[0] != count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_constant_i32 output length must equal count");
            }
            if (value < static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()) || value > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "fill_constant_i32 value is outside int32 range");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_fill_constant_i32(output.buffer->handle, static_cast<std::int32_t>(value), static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_fill_constant_i32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.fill_constant_i32");
    }, "Fill an int32 output tensor with a scalar value.\n\noutput is a rank-1 int32 tensor with length count. value must fit in the int32 range.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: int32\n- hip: int32\n- opencl: int32\n- vulkan: int32\n- metal: int32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("output"), nb::arg("value"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("fill_range", [](navatala_gpu::python::Tensor output, std::uint64_t count, std::int64_t start, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: output length must be count and start + count - 1 must fit in int32
        // target: operation the upstream generator.StdLib.DataFrame.Fill.fillRangeI32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (output.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "fill_range tensor is closed");
            }
            if (output.dtype_value != "int32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_range requires an int32 output tensor");
            }
            if (output.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_range expects a rank-1 output tensor");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "fill_range count exceeds uint32 index range");
            }
            if (output.shape_values[0] != count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_range output length must equal count");
            }
            if (start < static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()) || start > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "fill_range start is outside int32 range");
            }
            if (count > 0U) {
                const std::int64_t last_value = start + static_cast<std::int64_t>(count - 1U);
                if (last_value > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
                    throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "fill_range output values exceed int32 range");
                }
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_fill_range_i32(output.buffer->handle, static_cast<std::int32_t>(start), static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_fill_range_i32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.fill_range");
    }, "Fill an int32 DataFrame column with a sequential range.\n\noutput[i] = start + i for count rows. The binding rejects ranges that cannot be represented as int32 values.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: int32\n- hip: int32\n- opencl: int32\n- vulkan: int32\n- metal: int32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("output"), nb::arg("count"), nb::arg("start") = 0, nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("fill_validity_all_valid", [](navatala_gpu::python::Tensor output, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: output length must equal ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Fill.fillValidityAllValid
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (output.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "fill_validity_all_valid tensor is closed");
            }
            if (output.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_validity_all_valid requires a uint32 output tensor");
            }
            if (output.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_validity_all_valid expects a rank-1 output tensor");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "fill_validity_all_valid count exceeds uint32 index range");
            }
            const std::uint64_t validity_words = (count + 31U) / 32U;
            if (output.shape_values[0] != validity_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_validity_all_valid output length must equal ceil(count / 32)");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_fill_validity_all_valid_u32(output.buffer->handle, static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_fill_validity_all_valid_u32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.fill_validity_all_valid");
    }, "Fill a uint32 validity bitmap so all rows are valid.\n\noutput is a rank-1 uint32 tensor with ceil(count / 32) words. Unused trailing bits are zero.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: uint32\n- hip: uint32\n- opencl: uint32\n- vulkan: uint32\n- metal: uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("output"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_dataframe.def("fill_validity_all_null", [](navatala_gpu::python::Tensor output, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: output length must equal ceil(count / 32)
        // target: operation the upstream generator.StdLib.DataFrame.Fill.fillValidityAllNull
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (output.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "fill_validity_all_null tensor is closed");
            }
            if (output.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_validity_all_null requires a uint32 output tensor");
            }
            if (output.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_validity_all_null expects a rank-1 output tensor");
            }
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "fill_validity_all_null count exceeds uint32 index range");
            }
            const std::uint64_t validity_words = (count + 31U) / 32U;
            if (output.shape_values[0] != validity_words) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "fill_validity_all_null output length must equal ceil(count / 32)");
            }
            if (!backend.empty() && backend != "auto" && backend != output.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_dataframe_fill_validity_all_null_u32(output.buffer->handle, static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_dataframe_fill_validity_all_null_u32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, output);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return output;
        }, "dataframe.fill_validity_all_null");
    }, "Fill a uint32 validity bitmap so all rows are null.\n\noutput is a rank-1 uint32 tensor with ceil(count / 32) words.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: uint32\n- hip: uint32\n- opencl: uint32\n- vulkan: uint32\n- metal: uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("output"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_ml.def("mean_squared_error", [](navatala_gpu::python::Tensor y_true, navatala_gpu::python::Tensor y_pred, navatala_gpu::python::Tensor result, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: y_true and y_pred must have length count and count must be non-zero
        // target: operation the upstream generator.StdLib.RegressionMetrics.mseF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (y_true.is_closed() || y_pred.is_closed() || result.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "mean_squared_error tensor is closed");
            }
            if (y_true.dtype_value != "float32" || y_pred.dtype_value != "float32" || result.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_squared_error requires float32 tensors");
            }
            if (y_true.shape_values.size() != 1 || y_pred.shape_values.size() != 1 || !result.shape_values.empty()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_squared_error expects rank-1 inputs and scalar output");
            }
            if (count == 0U || count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_squared_error requires a non-empty uint32-sized count");
            }
            if (y_true.shape_values[0] != count || y_pred.shape_values[0] != count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_squared_error shape mismatch for count");
            }
            if (y_true.device_value != y_pred.device_value || y_true.device_value != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_squared_error requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_ml_mse_f32(y_true.buffer->handle, y_pred.buffer->handle, result.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_ml_mse_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_true);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_pred);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, result);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return result;
        }, "ml.mean_squared_error");
    }, "Compute float32 mean squared error.\n\ny_true and y_pred are rank-1 float32 tensors with length count. result is an explicit scalar float32 tensor.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32\n- hip: float32\n- opencl: float32\n- vulkan: float32\n- metal: float32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("y_true"), nb::arg("y_pred"), nb::arg("result"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_ml.def("root_mean_squared_error", [](navatala_gpu::python::Tensor y_true, navatala_gpu::python::Tensor y_pred, navatala_gpu::python::Tensor result, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: y_true and y_pred must have length count and count must be non-zero
        // target: operation the upstream generator.StdLib.RegressionMetrics.rmseF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (y_true.is_closed() || y_pred.is_closed() || result.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "root_mean_squared_error tensor is closed");
            }
            if (y_true.dtype_value != "float32" || y_pred.dtype_value != "float32" || result.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "root_mean_squared_error requires float32 tensors");
            }
            if (y_true.shape_values.size() != 1 || y_pred.shape_values.size() != 1 || !result.shape_values.empty()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "root_mean_squared_error expects rank-1 inputs and scalar output");
            }
            if (count == 0U || count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "root_mean_squared_error requires a non-empty uint32-sized count");
            }
            if (y_true.shape_values[0] != count || y_pred.shape_values[0] != count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "root_mean_squared_error shape mismatch for count");
            }
            if (y_true.device_value != y_pred.device_value || y_true.device_value != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "root_mean_squared_error requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_ml_rmse_f32(y_true.buffer->handle, y_pred.buffer->handle, result.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_ml_rmse_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_true);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_pred);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, result);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return result;
        }, "ml.root_mean_squared_error");
    }, "Compute float32 root mean squared error.\n\ny_true and y_pred are rank-1 float32 tensors with length count. result is an explicit scalar float32 tensor.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32\n- hip: float32\n- opencl: float32\n- vulkan: float32\n- metal: float32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("y_true"), nb::arg("y_pred"), nb::arg("result"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_ml.def("mean_absolute_error", [](navatala_gpu::python::Tensor y_true, navatala_gpu::python::Tensor y_pred, navatala_gpu::python::Tensor result, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: y_true and y_pred must have length count and count must be non-zero
        // target: operation the upstream generator.StdLib.RegressionMetrics.maeF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (y_true.is_closed() || y_pred.is_closed() || result.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "mean_absolute_error tensor is closed");
            }
            if (y_true.dtype_value != "float32" || y_pred.dtype_value != "float32" || result.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_absolute_error requires float32 tensors");
            }
            if (y_true.shape_values.size() != 1 || y_pred.shape_values.size() != 1 || !result.shape_values.empty()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_absolute_error expects rank-1 inputs and scalar output");
            }
            if (count == 0U || count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_absolute_error requires a non-empty uint32-sized count");
            }
            if (y_true.shape_values[0] != count || y_pred.shape_values[0] != count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_absolute_error shape mismatch for count");
            }
            if (y_true.device_value != y_pred.device_value || y_true.device_value != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_absolute_error requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_ml_mae_f32(y_true.buffer->handle, y_pred.buffer->handle, result.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_ml_mae_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_true);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_pred);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, result);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return result;
        }, "ml.mean_absolute_error");
    }, "Compute float32 mean absolute error.\n\ny_true and y_pred are rank-1 float32 tensors with length count. result is an explicit scalar float32 tensor.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32\n- hip: float32\n- opencl: float32\n- vulkan: float32\n- metal: float32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("y_true"), nb::arg("y_pred"), nb::arg("result"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_ml.def("mean_absolute_percentage_error", [](navatala_gpu::python::Tensor y_true, navatala_gpu::python::Tensor y_pred, navatala_gpu::python::Tensor result, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: y_true and y_pred must have length count and count must be non-zero
        // target: operation the upstream generator.StdLib.RegressionMetrics.mapeF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (y_true.is_closed() || y_pred.is_closed() || result.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "mean_absolute_percentage_error tensor is closed");
            }
            if (y_true.dtype_value != "float32" || y_pred.dtype_value != "float32" || result.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_absolute_percentage_error requires float32 tensors");
            }
            if (y_true.shape_values.size() != 1 || y_pred.shape_values.size() != 1 || !result.shape_values.empty()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_absolute_percentage_error expects rank-1 inputs and scalar output");
            }
            if (count == 0U || count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_absolute_percentage_error requires a non-empty uint32-sized count");
            }
            if (y_true.shape_values[0] != count || y_pred.shape_values[0] != count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_absolute_percentage_error shape mismatch for count");
            }
            if (y_true.device_value != y_pred.device_value || y_true.device_value != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "mean_absolute_percentage_error requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_ml_mape_f32(y_true.buffer->handle, y_pred.buffer->handle, result.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_ml_mape_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_true);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_pred);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, result);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return result;
        }, "ml.mean_absolute_percentage_error");
    }, "Compute float32 mean absolute percentage error.\n\ny_true and y_pred are rank-1 float32 tensors with length count. result is an explicit scalar float32 tensor. y_true == 0 follows IEEE-754 division semantics.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32\n- hip: float32\n- opencl: float32\n- vulkan: float32\n- metal: float32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("y_true"), nb::arg("y_pred"), nb::arg("result"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_ml.def("r2_score", [](navatala_gpu::python::Tensor y_true, navatala_gpu::python::Tensor y_pred, navatala_gpu::python::Tensor y_mean, navatala_gpu::python::Tensor result, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: y_true and y_pred must have length count; y_mean/result must be scalar float32 tensors; count must be non-zero
        // target: operation the upstream generator.StdLib.RegressionMetrics.r2ScoreF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (y_true.is_closed() || y_pred.is_closed() || y_mean.is_closed() || result.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "r2_score tensor is closed");
            }
            if (y_true.dtype_value != "float32" || y_pred.dtype_value != "float32" || y_mean.dtype_value != "float32" || result.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "r2_score requires float32 tensors");
            }
            if (y_true.shape_values.size() != 1 || y_pred.shape_values.size() != 1 || !y_mean.shape_values.empty() || !result.shape_values.empty()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "r2_score expects rank-1 inputs plus scalar y_mean/result tensors");
            }
            if (count == 0U || count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "r2_score requires a non-empty uint32-sized count");
            }
            if (y_true.shape_values[0] != count || y_pred.shape_values[0] != count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "r2_score shape mismatch for count");
            }
            if (y_true.device_value != y_pred.device_value || y_true.device_value != y_mean.device_value || y_true.device_value != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "r2_score requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_ml_r2_score_f32(y_true.buffer->handle, y_pred.buffer->handle, y_mean.buffer->handle, result.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_ml_r2_score_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_true);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_pred);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_mean);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, result);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return result;
        }, "ml.r2_score");
    }, "Compute float32 R2 score.\n\ny_true and y_pred are rank-1 float32 tensors with length count. y_mean is an explicit scalar float32 tensor containing the precomputed mean of y_true; result is an explicit scalar float32 tensor. Constant y_true inputs follow IEEE-754 division semantics.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32\n- hip: float32\n- opencl: float32\n- vulkan: float32\n- metal: float32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("y_true"), nb::arg("y_pred"), nb::arg("y_mean"), nb::arg("result"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_ml.def("explained_variance_score", [](navatala_gpu::python::Tensor y_true, navatala_gpu::python::Tensor y_pred, navatala_gpu::python::Tensor mean_residuals, navatala_gpu::python::Tensor mean_true, navatala_gpu::python::Tensor result, std::uint64_t count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: y_true and y_pred must have length count; mean_residuals/mean_true/result must be scalar float32 tensors; count must be non-zero
        // target: operation the upstream generator.StdLib.RegressionMetrics.explainedVarianceF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (y_true.is_closed() || y_pred.is_closed() || mean_residuals.is_closed() || mean_true.is_closed() || result.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "explained_variance_score tensor is closed");
            }
            if (y_true.dtype_value != "float32" || y_pred.dtype_value != "float32" || mean_residuals.dtype_value != "float32" || mean_true.dtype_value != "float32" || result.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "explained_variance_score requires float32 tensors");
            }
            if (y_true.shape_values.size() != 1 || y_pred.shape_values.size() != 1 || !mean_residuals.shape_values.empty() || !mean_true.shape_values.empty() || !result.shape_values.empty()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "explained_variance_score expects rank-1 inputs plus scalar mean/result tensors");
            }
            if (count == 0U || count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "explained_variance_score requires a non-empty uint32-sized count");
            }
            if (y_true.shape_values[0] != count || y_pred.shape_values[0] != count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "explained_variance_score shape mismatch for count");
            }
            if (y_true.device_value != y_pred.device_value || y_true.device_value != mean_residuals.device_value || y_true.device_value != mean_true.device_value || y_true.device_value != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "explained_variance_score requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != result.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_ml_explained_variance_score_f32(y_true.buffer->handle, y_pred.buffer->handle, mean_residuals.buffer->handle, mean_true.buffer->handle, result.buffer->handle,
                    static_cast<std::size_t>(count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_ml_explained_variance_score_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_true);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y_pred);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, mean_residuals);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, mean_true);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, result);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return result;
        }, "ml.explained_variance_score");
    }, "Compute float32 explained variance score.\n\ny_true and y_pred are rank-1 float32 tensors with length count. mean_residuals and mean_true are explicit scalar float32 tensors; result is an explicit scalar float32 tensor. Constant y_true inputs follow IEEE-754 division semantics.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32\n- hip: float32\n- opencl: float32\n- vulkan: float32\n- metal: float32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("y_true"), nb::arg("y_pred"), nb::arg("mean_residuals"), nb::arg("mean_true"), nb::arg("result"), nb::arg("count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_sparse.def("csr_spmv", [](navatala_gpu::python::Tensor row_offsets, navatala_gpu::python::Tensor col_indices, navatala_gpu::python::Tensor values, navatala_gpu::python::Tensor x, navatala_gpu::python::Tensor y, std::uint64_t rows, std::uint64_t cols, std::uint64_t nnz, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: row_offsets length must be rows + 1; col_indices and values length must be nnz; x/y lengths must match cols/rows
        // target: operation the upstream generator.StdLib.LinearSolver.SpMV.csrSpMVF32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (row_offsets.is_closed() || col_indices.is_closed() || values.is_closed() || x.is_closed() || y.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "csr_spmv tensor is closed");
            }
            if (row_offsets.dtype_value != "uint32" || col_indices.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "csr_spmv requires uint32 row_offsets and col_indices");
            }
            if (values.dtype_value != "float32" || x.dtype_value != "float32" || y.dtype_value != "float32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "csr_spmv requires float32 values, x, and y");
            }
            if (row_offsets.shape_values.size() != 1 || col_indices.shape_values.size() != 1 || values.shape_values.size() != 1 || x.shape_values.size() != 1 || y.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "csr_spmv requires rank-1 CSR buffers and vectors");
            }
            if (rows == std::numeric_limits<std::uint64_t>::max()) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "csr_spmv rows overflow row_offsets length");
            }
            const std::uint64_t expected_row_offsets = rows + 1;
            if (expected_row_offsets == 0 || row_offsets.shape_values[0] != expected_row_offsets || col_indices.shape_values[0] != nnz || values.shape_values[0] != nnz || x.shape_values[0] != cols || y.shape_values[0] != rows) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "csr_spmv shape mismatch for rows/cols/nnz");
            }
            if (row_offsets.device_value != col_indices.device_value || row_offsets.device_value != values.device_value || row_offsets.device_value != x.device_value || row_offsets.device_value != y.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "csr_spmv requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != y.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            const std::uint64_t maxSize = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
            if (rows > maxSize || cols > maxSize || nnz > maxSize) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "csr_spmv dimensions overflow size_t");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_csr_spmv_f32(row_offsets.buffer->handle, col_indices.buffer->handle, values.buffer->handle, x.buffer->handle, y.buffer->handle,
                    static_cast<std::size_t>(rows), static_cast<std::size_t>(cols), static_cast<std::size_t>(nnz), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_csr_spmv_f32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, row_offsets);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, col_indices);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, values);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, x);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, y);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return y;
        }, "sparse.csr_spmv");
    }, "Compute y = A * x for a float32 CSR sparse matrix.\n\nThe CSR descriptor uses uint32 row_offsets/col_indices and float32 values. rows, cols, and nnz make the descriptor contract explicit at the Python boundary.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: float32, uint32\n- hip: float32, uint32\n- opencl: float32, uint32\n- vulkan: float32, uint32\n- metal: float32, uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("row_offsets"), nb::arg("col_indices"), nb::arg("values"), nb::arg("x"), nb::arg("y"), nb::arg("rows"), nb::arg("cols"), nb::arg("nnz"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_graph.def("out_degree", [](navatala_gpu::python::Tensor row_offsets, navatala_gpu::python::Tensor degrees, std::uint64_t vertex_count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: row_offsets length must be vertex_count + 1 and degrees length must be vertex_count
        // target: operation the upstream generator.StdLib.Graph.outDegreeU32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (row_offsets.is_closed() || degrees.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "out_degree tensor is closed");
            }
            if (row_offsets.dtype_value != "uint32" || degrees.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "out_degree requires uint32 tensors");
            }
            if (row_offsets.shape_values.size() != 1 || degrees.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "out_degree requires rank-1 tensors");
            }
            if (vertex_count == std::numeric_limits<std::uint64_t>::max()) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "out_degree vertex_count overflows row_offsets length");
            }
            const std::uint64_t expected_row_offsets = vertex_count + 1;
            if (expected_row_offsets == 0 || row_offsets.shape_values[0] != expected_row_offsets || degrees.shape_values[0] != vertex_count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "out_degree shape mismatch for vertex_count");
            }
            if (row_offsets.device_value != degrees.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "out_degree requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != degrees.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            const std::uint64_t maxSize = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
            if (vertex_count > maxSize) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "out_degree vertex_count overflows size_t");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_graph_out_degree_u32(row_offsets.buffer->handle, degrees.buffer->handle, static_cast<std::size_t>(vertex_count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_graph_out_degree_u32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, row_offsets);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, degrees);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return degrees;
        }, "graph.out_degree");
    }, "Compute per-vertex out-degree from CSR row offsets.\n\nThe graph descriptor uses uint32 row_offsets with length vertex_count + 1. The output degrees tensor has length vertex_count.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: uint32\n- hip: uint32\n- opencl: uint32\n- vulkan: uint32\n- metal: uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("row_offsets"), nb::arg("degrees"), nb::arg("vertex_count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
    m_graph.def("in_degree", [](navatala_gpu::python::Tensor col_offsets, navatala_gpu::python::Tensor degrees, std::uint64_t vertex_count, std::string backend, nb::object queue, bool sync) {
        // precondition[invalid_shape]: col_offsets length must be vertex_count + 1 and degrees length must be vertex_count
        // target: operation the upstream generator.StdLib.Graph.inDegreeU32
        return navatala_gpu::python::translate_status([&]() -> navatala_gpu::python::Tensor {
            if (col_offsets.is_closed() || degrees.is_closed()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_HANDLE, "in_degree tensor is closed");
            }
            if (col_offsets.dtype_value != "uint32" || degrees.dtype_value != "uint32") {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "in_degree requires uint32 tensors");
            }
            if (col_offsets.shape_values.size() != 1 || degrees.shape_values.size() != 1) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "in_degree requires rank-1 tensors");
            }
            if (vertex_count == std::numeric_limits<std::uint64_t>::max()) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "in_degree vertex_count overflows col_offsets length");
            }
            const std::uint64_t expected_col_offsets = vertex_count + 1;
            if (expected_col_offsets == 0 || col_offsets.shape_values[0] != expected_col_offsets || degrees.shape_values[0] != vertex_count) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "in_degree shape mismatch for vertex_count");
            }
            if (col_offsets.device_value != degrees.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "in_degree requires tensors on the same backend");
            }
            if (!backend.empty() && backend != "auto" && backend != degrees.device_value) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "requested backend does not match tensor backend");
            }
            const std::uint64_t maxSize = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
            if (vertex_count > maxSize) {
                throw navatala_gpu::python::StatusError(NAVATALA_OVERFLOW_ERROR, "in_degree vertex_count overflows size_t");
            }
            if (sync && queue.is_none()) {
                throw navatala_gpu::python::StatusError(NAVATALA_INVALID_PARAM, "sync=True requires an explicit Queue");
            }
            navatala_gpu::python::check_status(
                navatala_gpu_graph_in_degree_u32(col_offsets.buffer->handle, degrees.buffer->handle, static_cast<std::size_t>(vertex_count), navatala_gpu::python::queue_handle_from_object(queue)),
                "navatala_gpu_graph_in_degree_u32");
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, col_offsets);
            navatala_gpu::python::retain_tensor_until_queue_sync(queue, degrees);
            if (sync) {
                navatala_gpu::python::sync_queue_object(queue);
            }
            return degrees;
        }, "graph.in_degree");
    }, "Compute per-vertex in-degree from CSC column offsets.\n\nThe graph descriptor uses uint32 CSC/transposed-CSR col_offsets with length vertex_count + 1. The output degrees tensor has length vertex_count.\n\nMaturity: experimental\n\nSupported backend/dtype pairs:\n- cuda: uint32\n- hip: uint32\n- opencl: uint32\n- vulkan: uint32\n- metal: uint32\n\nSynchronization:\nHost synchronization is never inserted by default.", nb::arg("col_offsets"), nb::arg("degrees"), nb::arg("vertex_count"), nb::arg("backend") = "auto", nb::kw_only(), nb::arg("queue") = nb::none(), nb::arg("sync") = false);
}