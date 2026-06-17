// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "navatala/navatala_ffi.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static_assert(NAVATALA_GPU_FFI_ABI_VERSION == 6, "runtime_helpers.hpp is incompatible with this navatala_ffi.h ABI version");

namespace navatala_gpu::python {

class StatusError final : public std::runtime_error {
public:
    StatusError(NavatalaErrorCode code, const std::string& operation)
        : std::runtime_error(operation), code_(code) {}

    NavatalaErrorCode code() const noexcept { return code_; }

private:
    NavatalaErrorCode code_;
};

inline void check_status(NavatalaErrorCode code, const std::string& operation) {
    if (code != NAVATALA_SUCCESS) {
        throw StatusError(code, operation);
    }
}

inline NavatalaBackend parse_backend(const std::string& backend) {
    if (backend == "cuda") return NAVATALA_BACKEND_CUDA_FFI;
    if (backend == "hip") return NAVATALA_BACKEND_HIP_FFI;
    if (backend == "metal") return NAVATALA_BACKEND_METAL_FFI;
    if (backend == "opencl") return NAVATALA_BACKEND_OPENCL_FFI;
    if (backend == "vulkan") return NAVATALA_BACKEND_VULKAN_FFI;
    if (backend == "auto" || backend.empty()) return NAVATALA_BACKEND_AUTO_FFI;
    throw StatusError(NAVATALA_INVALID_PARAM, "unknown backend: " + backend);
}

inline std::string backend_name(NavatalaBackend backend) {
    switch (backend) {
        case NAVATALA_BACKEND_CUDA_FFI: return "cuda";
        case NAVATALA_BACKEND_HIP_FFI: return "hip";
        case NAVATALA_BACKEND_METAL_FFI: return "metal";
        case NAVATALA_BACKEND_OPENCL_FFI: return "opencl";
        case NAVATALA_BACKEND_VULKAN_FFI: return "vulkan";
        case NAVATALA_BACKEND_AUTO_FFI: return "auto";
        default: return "unknown";
    }
}

inline std::size_t dtype_size(const std::string& dtype) {
    if (dtype == "float16" || dtype == "Float16" || dtype == "bfloat16" || dtype == "BFloat16" || dtype == "bf16") return 2;
    if (dtype == "float32" || dtype == "Float32" || dtype == "int32" || dtype == "Int32" || dtype == "uint32" || dtype == "UInt32") return 4;
    if (dtype == "float64" || dtype == "Float64" || dtype == "int64" || dtype == "Int64" || dtype == "uint64" || dtype == "UInt64") return 8;
    if (dtype == "int16" || dtype == "Int16" || dtype == "uint16" || dtype == "UInt16") return 2;
    if (dtype == "int8" || dtype == "Int8" || dtype == "uint8" || dtype == "UInt8" || dtype == "bool" || dtype == "Bool") return 1;
    throw StatusError(NAVATALA_INVALID_PARAM, "unsupported dtype: " + dtype);
}

inline std::uint64_t checked_element_count(const std::vector<std::uint64_t>& shape) {
    std::uint64_t count = 1;
    for (std::uint64_t dim : shape) {
        if (dim != 0 && count > std::numeric_limits<std::uint64_t>::max() / dim) {
            throw StatusError(NAVATALA_OVERFLOW_ERROR, "tensor shape overflows uint64 element count");
        }
        count *= dim;
    }
    return count;
}

struct ContextControlBlock {
    NavatalaGpuContext* handle = nullptr;
    NavatalaBackend backend = NAVATALA_BACKEND_AUTO_FFI;
    int device_id = 0;

    ContextControlBlock(NavatalaBackend requested_backend, int requested_device_id)
        : backend(requested_backend), device_id(requested_device_id) {
        check_status(navatala_gpu_create_context(requested_backend, requested_device_id, &handle), "navatala_gpu_create_context");
        if (handle != nullptr) {
            backend = navatala_gpu_context_get_backend(handle);
            device_id = navatala_gpu_context_get_device_id(handle);
        }
    }

    ~ContextControlBlock() noexcept {
        if (handle != nullptr) {
            navatala_gpu_destroy_context(handle);
        }
    }

    ContextControlBlock(const ContextControlBlock&) = delete;
    ContextControlBlock& operator=(const ContextControlBlock&) = delete;
};

struct BufferControlBlock {
    std::shared_ptr<ContextControlBlock> context;
    std::shared_ptr<void> external_owner;
    NavatalaGpuBuffer* handle = nullptr;
    std::size_t bytes = 0;
    std::atomic<std::uint64_t> live_dlpack_exports{0};

    BufferControlBlock(std::shared_ptr<ContextControlBlock> ctx, std::size_t requested_bytes)
        : context(std::move(ctx)), bytes(requested_bytes) {
        if (!context || context->handle == nullptr) {
            throw StatusError(NAVATALA_INVALID_HANDLE, "buffer context is invalid");
        }
        check_status(navatala_gpu_create_buffer(context->handle, requested_bytes, &handle), "navatala_gpu_create_buffer");
    }

    BufferControlBlock(std::shared_ptr<ContextControlBlock> ctx, void* device_ptr, std::size_t requested_bytes, std::shared_ptr<void> owner)
        : context(std::move(ctx)), external_owner(std::move(owner)), bytes(requested_bytes) {
        if (!context || context->handle == nullptr) {
            throw StatusError(NAVATALA_INVALID_HANDLE, "buffer context is invalid");
        }
        if (device_ptr == nullptr && requested_bytes != 0) {
            throw StatusError(NAVATALA_INVALID_PARAM, "DLPack tensor has null device pointer");
        }
        check_status(navatala_gpu_wrap_device_ptr(context->handle, device_ptr, requested_bytes, &handle), "navatala_gpu_wrap_device_ptr");
    }

    ~BufferControlBlock() noexcept {
        if (handle != nullptr) {
            navatala_gpu_destroy_buffer(handle);
        }
    }

    BufferControlBlock(const BufferControlBlock&) = delete;
    BufferControlBlock& operator=(const BufferControlBlock&) = delete;
};

struct Buffer {
    std::shared_ptr<BufferControlBlock> buffer;
    std::string device_value;

    Buffer() = default;
    Buffer(std::shared_ptr<BufferControlBlock> b, std::string device)
        : buffer(std::move(b)), device_value(std::move(device)) {}

    std::string device() const { return device_value; }
    bool is_closed() const noexcept { return buffer == nullptr; }
    std::size_t nbytes() const noexcept { return buffer ? buffer->bytes : 0; }
    void close() {
        if (buffer != nullptr && buffer->live_dlpack_exports.load(std::memory_order_acquire) != 0) {
            throw StatusError(NAVATALA_RUNTIME_ERROR, "buffer has live DLPack exports; release or consume exported capsules before close");
        }
        buffer.reset();
    }
};

struct QueueControlBlock {
    std::shared_ptr<ContextControlBlock> context;
    NavatalaGpuQueue* handle = nullptr;
    std::mutex retained_buffers_mutex;
    std::vector<std::shared_ptr<BufferControlBlock>> retained_buffers;

    QueueControlBlock(std::shared_ptr<ContextControlBlock> ctx, int priority)
        : context(std::move(ctx)) {
        if (!context || context->handle == nullptr) {
            throw StatusError(NAVATALA_INVALID_HANDLE, "queue context is invalid");
        }
        check_status(navatala_gpu_create_queue(context->handle, priority, &handle), "navatala_gpu_create_queue");
    }

    ~QueueControlBlock() noexcept {
        if (handle != nullptr) {
            (void) navatala_gpu_queue_sync(handle);
            navatala_gpu_destroy_queue(handle);
        }
        release_retained_buffers_noexcept();
    }

    void retain_buffer(std::shared_ptr<BufferControlBlock> buffer) {
        if (!buffer) {
            return;
        }
        std::lock_guard<std::mutex> lock(retained_buffers_mutex);
        retained_buffers.push_back(std::move(buffer));
    }

    void release_retained_buffers_noexcept() noexcept {
        try {
            std::lock_guard<std::mutex> lock(retained_buffers_mutex);
            retained_buffers.clear();
        } catch (...) {
        }
    }

    void synchronize_and_release() {
        if (handle == nullptr) {
            release_retained_buffers_noexcept();
            return;
        }
        check_status(navatala_gpu_queue_sync(handle), "navatala_gpu_queue_sync");
        release_retained_buffers_noexcept();
    }

    QueueControlBlock(const QueueControlBlock&) = delete;
    QueueControlBlock& operator=(const QueueControlBlock&) = delete;
};

struct EventControlBlock {
    std::shared_ptr<ContextControlBlock> context;
    NavatalaGpuEvent* handle = nullptr;

    explicit EventControlBlock(std::shared_ptr<ContextControlBlock> ctx)
        : context(std::move(ctx)) {
        if (!context || context->handle == nullptr) {
            throw StatusError(NAVATALA_INVALID_HANDLE, "event context is invalid");
        }
        check_status(navatala_gpu_create_event(context->handle, &handle), "navatala_gpu_create_event");
    }

    ~EventControlBlock() noexcept {
        if (handle != nullptr) {
            navatala_gpu_destroy_event(handle);
        }
    }

    EventControlBlock(const EventControlBlock&) = delete;
    EventControlBlock& operator=(const EventControlBlock&) = delete;
};

struct Tensor {
    std::shared_ptr<BufferControlBlock> buffer;
    std::vector<std::uint64_t> shape_values;
    std::string dtype_value;
    std::string device_value;
    bool readonly = false;

    Tensor() = default;
    Tensor(std::shared_ptr<BufferControlBlock> b, std::vector<std::uint64_t> shape, std::string dtype, std::string device)
        : buffer(std::move(b)), shape_values(std::move(shape)), dtype_value(std::move(dtype)), device_value(std::move(device)) {}

    std::vector<std::uint64_t> shape() const { return shape_values; }
    std::string dtype() const { return dtype_value; }
    std::string device() const { return device_value; }
    bool is_closed() const noexcept { return buffer == nullptr; }
    std::size_t nbytes() const noexcept { return buffer ? buffer->bytes : 0; }
    void close() {
        if (buffer != nullptr && buffer->live_dlpack_exports.load(std::memory_order_acquire) != 0) {
            throw StatusError(NAVATALA_RUNTIME_ERROR, "tensor has live DLPack exports; release or consume exported capsules before close");
        }
        buffer.reset();
    }
};

inline Buffer allocate_buffer(std::size_t bytes, NavatalaBackend backend, int device_id) {
    auto context = std::make_shared<ContextControlBlock>(backend, device_id);
    auto buffer = std::make_shared<BufferControlBlock>(context, bytes);
    return Buffer(std::move(buffer), backend_name(context->backend));
}

inline Tensor allocate_tensor(const std::vector<std::uint64_t>& shape, const std::string& dtype, NavatalaBackend backend, int device_id) {
    std::uint64_t elements = checked_element_count(shape);
    std::size_t itemsize = dtype_size(dtype);
    if (elements > std::numeric_limits<std::size_t>::max() / itemsize) {
        throw StatusError(NAVATALA_OVERFLOW_ERROR, "tensor allocation size overflows size_t");
    }
    std::size_t bytes = static_cast<std::size_t>(elements) * itemsize;
    auto context = std::make_shared<ContextControlBlock>(backend, device_id);
    auto buffer = std::make_shared<BufferControlBlock>(context, bytes);
    return Tensor(std::move(buffer), shape, dtype, backend_name(context->backend));
}

} // namespace navatala_gpu::python
