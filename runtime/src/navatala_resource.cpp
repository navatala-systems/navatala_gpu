// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Navatala C++ Wrapper Layer - Resource Implementation
// Part of the Navatala GPU Runtime Library
#include "../include/navatala/resource.hpp"
#include "../include/navatala/navatala_ffi.h"

namespace navatala {

// ============================================================================
// Queue Implementation
// ============================================================================

queue::~queue() noexcept {
    if (handle_) {
        navatala_gpu_destroy_queue(handle_);
        handle_ = nullptr;
    }
}

void queue::sync() const {
    if (!handle_) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw exception(error_code::invalid_handle, "sync() called on invalid queue");
#else
        return;
#endif
    }

    NavatalaErrorCode err = navatala_gpu_queue_sync(handle_);
    if (err != NAVATALA_SUCCESS) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw gpu_runtime_error(translate_error(err), "queue::sync()");
#endif
    }
}

status queue::sync_status() const noexcept {
    if (!handle_) {
        return status(error_code::invalid_handle, "sync() called on invalid queue");
    }

    NavatalaErrorCode err = navatala_gpu_queue_sync(handle_);
    if (err != NAVATALA_SUCCESS) {
        return status(translate_error(err), "queue::sync()");
    }
    return status();
}

bool queue::is_ready() const noexcept {
    if (!handle_) {
        return true;  // Invalid queue is "ready" (no work)
    }
    return navatala_gpu_queue_is_ready(handle_) != 0;
}

backend queue::get_backend() const noexcept {
    if (!handle_) {
        return backend_;
    }
    NavatalaBackend b = navatala_gpu_queue_get_backend(handle_);
    switch (b) {
        case NAVATALA_BACKEND_CUDA_FFI:   return backend::cuda;
        case NAVATALA_BACKEND_HIP_FFI:    return backend::hip;
        case NAVATALA_BACKEND_METAL_FFI:  return backend::metal;
        case NAVATALA_BACKEND_OPENCL_FFI: return backend::opencl;
        case NAVATALA_BACKEND_VULKAN_FFI: return backend::vulkan;
        default:                          return backend::auto_select;
    }
}

void queue::record(event& e) const {
    if (!handle_ || !e.handle_) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw exception(error_code::invalid_handle, "record() with invalid queue or event");
#else
        return;
#endif
    }

    NavatalaErrorCode err = navatala_gpu_event_record(e.handle_, handle_);
    if (err != NAVATALA_SUCCESS) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw gpu_runtime_error(translate_error(err), "queue::record()");
#endif
    }
    e.recorded_ = true;
}

void queue::wait_for(const event& e) {
    if (!handle_ || !e.handle_) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw exception(error_code::invalid_handle, "wait_for() with invalid queue or event");
#else
        return;
#endif
    }

    NavatalaErrorCode err = navatala_gpu_event_wait(handle_, e.handle_);
    if (err != NAVATALA_SUCCESS) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw gpu_runtime_error(translate_error(err), "queue::wait_for()");
#endif
    }
}

// ============================================================================
// Event Implementation
// ============================================================================

result<event> event::create(resources& res) {
    if (!res.handle()) {
        return result<event>(error_code::invalid_handle, "create() with invalid resources");
    }

    NavatalaGpuEvent* evt = nullptr;
    NavatalaErrorCode err = navatala_gpu_create_event(res.handle(), &evt);
    if (err != NAVATALA_SUCCESS) {
        return result<event>(translate_error(err), "event::create()");
    }

    return event(evt, res.handle());
}

event::~event() noexcept {
    if (handle_) {
        navatala_gpu_destroy_event(handle_);
        handle_ = nullptr;
    }
}

void event::record(const queue& q) {
    if (!handle_ || !q.handle()) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw exception(error_code::invalid_handle, "record() with invalid event or queue");
#else
        return;
#endif
    }

    NavatalaErrorCode err = navatala_gpu_event_record(handle_, q.handle());
    if (err != NAVATALA_SUCCESS) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw gpu_runtime_error(translate_error(err), "event::record()");
#endif
    }
    recorded_ = true;
}

status event::record_status(const queue& q) noexcept {
    if (!handle_ || !q.handle()) {
        return status(error_code::invalid_handle, "record() with invalid event or queue");
    }

    NavatalaErrorCode err = navatala_gpu_event_record(handle_, q.handle());
    if (err != NAVATALA_SUCCESS) {
        return status(translate_error(err), "event::record()");
    }
    recorded_ = true;
    return status();
}

void event::stream_wait(queue& q) const {
    if (!handle_ || !q.handle()) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw exception(error_code::invalid_handle, "stream_wait() with invalid event or queue");
#else
        return;
#endif
    }

    NavatalaErrorCode err = navatala_gpu_event_wait(q.handle(), handle_);
    if (err != NAVATALA_SUCCESS) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw gpu_runtime_error(translate_error(err), "event::stream_wait()");
#endif
    }
}

status event::stream_wait_status(queue& q) const noexcept {
    if (!handle_ || !q.handle()) {
        return status(error_code::invalid_handle, "stream_wait() with invalid event or queue");
    }

    NavatalaErrorCode err = navatala_gpu_event_wait(q.handle(), handle_);
    if (err != NAVATALA_SUCCESS) {
        return status(translate_error(err), "event::stream_wait()");
    }
    return status();
}

void event::synchronize() const {
    if (!handle_) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw exception(error_code::invalid_handle, "synchronize() on invalid event");
#else
        return;
#endif
    }

    NavatalaErrorCode err = navatala_gpu_event_synchronize(handle_);
    if (err != NAVATALA_SUCCESS) {
#if NAVATALA_ENABLE_EXCEPTIONS
        throw gpu_runtime_error(translate_error(err), "event::synchronize()");
#endif
    }
}

status event::synchronize_status() const noexcept {
    if (!handle_) {
        return status(error_code::invalid_handle, "synchronize() on invalid event");
    }

    NavatalaErrorCode err = navatala_gpu_event_synchronize(handle_);
    if (err != NAVATALA_SUCCESS) {
        return status(translate_error(err), "event::synchronize()");
    }
    return status();
}

bool event::is_complete() const noexcept {
    if (!handle_) {
        return true;  // Invalid event is "complete"
    }
    return navatala_gpu_event_is_complete(handle_) != 0;
}

// ============================================================================
// Resources Implementation
// ============================================================================

static NavatalaBackend backend_to_ffi(backend b) {
    switch (b) {
        case backend::cuda:        return NAVATALA_BACKEND_CUDA_FFI;
        case backend::hip:         return NAVATALA_BACKEND_HIP_FFI;
        case backend::metal:       return NAVATALA_BACKEND_METAL_FFI;
        case backend::opencl:      return NAVATALA_BACKEND_OPENCL_FFI;
        case backend::vulkan:      return NAVATALA_BACKEND_VULKAN_FFI;
        case backend::auto_select: return NAVATALA_BACKEND_AUTO_FFI;
        default:                   return NAVATALA_BACKEND_AUTO_FFI;
    }
}

static backend ffi_to_backend(NavatalaBackend b) {
    switch (b) {
        case NAVATALA_BACKEND_CUDA_FFI:   return backend::cuda;
        case NAVATALA_BACKEND_HIP_FFI:    return backend::hip;
        case NAVATALA_BACKEND_METAL_FFI:  return backend::metal;
        case NAVATALA_BACKEND_OPENCL_FFI: return backend::opencl;
        case NAVATALA_BACKEND_VULKAN_FFI: return backend::vulkan;
        case NAVATALA_BACKEND_AUTO_FFI:   return backend::auto_select;
        default:                          return backend::auto_select;
    }
}

result<resources> resources::create(backend b) {
    return create(0, b);
}

result<resources> resources::create(int device_id, backend b) {
    // Create context
    NavatalaGpuContext* ctx = nullptr;
    NavatalaErrorCode err = navatala_gpu_create_context(backend_to_ffi(b), device_id, &ctx);
    if (err != NAVATALA_SUCCESS) {
        return result<resources>(translate_error(err), "resources::create() context");
    }

    // Determine actual backend
    backend actual_backend = ffi_to_backend(navatala_gpu_context_get_backend(ctx));

    // Create default queue
    NavatalaGpuQueue* q = nullptr;
    err = navatala_gpu_create_queue(ctx, static_cast<int>(queue::priority::normal), &q);
    if (err != NAVATALA_SUCCESS) {
        navatala_gpu_destroy_context(ctx);
        return result<resources>(translate_error(err), "resources::create() queue");
    }

    queue default_q(q, ctx, actual_backend, queue::priority::normal);

    // Create default memory pool
    NavatalaGpuMemoryResource* pool_handle = nullptr;
    err = navatala_gpu_create_memory_pool(ctx,
                                          NAVATALA_DEFAULT_POOL_INITIAL_SIZE,
                                          NAVATALA_DEFAULT_POOL_MAX_SIZE,
                                          &pool_handle);
    if (err != NAVATALA_SUCCESS) {
        // Queue destructor will be called, context needs manual cleanup
        navatala_gpu_destroy_context(ctx);
        return result<resources>(translate_error(err), "resources::create() memory pool");
    }

    auto pool = std::make_unique<memory::pool>(
        memory::pool(pool_handle, ctx,
                     NAVATALA_DEFAULT_POOL_INITIAL_SIZE,
                     NAVATALA_DEFAULT_POOL_MAX_SIZE));

    return resources(ctx, actual_backend, device_id, std::move(default_q), std::move(pool));
}

resources::~resources() noexcept {
    // Memory pool will be destroyed by unique_ptr
    // Default queue destructor handles its cleanup

    if (handle_) {
        // Destroy context last
        navatala_gpu_destroy_context(handle_);
        handle_ = nullptr;
    }
}

result<queue> resources::create_queue(queue::priority priority) {
    if (!handle_) {
        return result<queue>(error_code::invalid_handle, "create_queue() on invalid resources");
    }

    NavatalaGpuQueue* q = nullptr;
    NavatalaErrorCode err = navatala_gpu_create_queue(handle_, static_cast<int>(priority), &q);
    if (err != NAVATALA_SUCCESS) {
        return result<queue>(translate_error(err), "resources::create_queue()");
    }

    return queue(q, handle_, backend_, priority);
}

result<event> resources::create_event() {
    return event::create(*this);
}

result<memory::pool> resources::create_memory_pool(size_t initial_size, size_t max_size) {
    if (!handle_) {
        return result<memory::pool>(error_code::invalid_handle,
                                    "create_memory_pool() on invalid resources");
    }

    NavatalaGpuMemoryResource* pool_handle = nullptr;
    NavatalaErrorCode err = navatala_gpu_create_memory_pool(handle_, initial_size, max_size, &pool_handle);
    if (err != NAVATALA_SUCCESS) {
        return result<memory::pool>(translate_error(err), "resources::create_memory_pool()");
    }

    return memory::pool(pool_handle, handle_, initial_size, max_size);
}

} // namespace navatala
