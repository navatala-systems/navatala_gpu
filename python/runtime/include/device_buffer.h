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

// Device Buffer - RAII typed device memory buffer
// Part of the RMM formalization effort
//
// Provides a type-safe RAII wrapper around GPU memory allocation.
// Supports move semantics but not copy (to prevent double-free).

#pragma once

#include <gpu_runtime.h>
#include <memory>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <algorithm>

namespace GpuRuntime {

/// RAII device memory buffer with typed access.
///
/// Manages GPU memory allocation and deallocation using RAII.
/// Supports move semantics but not copy.
///
/// This implementation uses the gpu_runtime Buffer class directly
/// for proper integration with Queue::memcpy operations.
///
/// Example usage:
///   DeviceBuffer<float> buffer(device, 1024);  // Allocate 1024 floats
///   buffer.copy_from_host(host_data.data(), 1024, queue);
///   // ... use buffer ...
///   // Memory automatically freed when buffer goes out of scope
template<typename T>
class DeviceBuffer {
public:
    static_assert(std::is_trivially_copyable<T>::value,
                  "DeviceBuffer only supports trivially copyable types");

    /// Default constructor - creates empty buffer
    DeviceBuffer() = default;

    /// Construct buffer with given number of elements.
    /// @param device Device to allocate on
    /// @param count Number of elements (NOT bytes)
    /// @param kind Memory kind (default: Device)
    DeviceBuffer(Device& device, size_t count, MemoryKind kind = MemoryKind::Device)
        : device_(&device)
        , count_(count)
    {
        if (count_ > 0) {
            buffer_ = device.createBuffer(size_bytes(), kind);
            if (!buffer_) {
                throw std::runtime_error("Failed to allocate device buffer");
            }
        }
    }

    /// Move constructor (transfers ownership)
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : device_(other.device_)
        , buffer_(std::move(other.buffer_))
        , count_(other.count_)
    {
        other.count_ = 0;
    }

    /// Move assignment (transfers ownership)
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            device_ = other.device_;
            buffer_ = std::move(other.buffer_);
            count_ = other.count_;
            other.count_ = 0;
        }
        return *this;
    }

    /// No copy (RAII)
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /// Destructor - buffer automatically freed by unique_ptr
    ~DeviceBuffer() = default;

    /// Get pointer to device memory
    T* data() noexcept {
        return buffer_ ? static_cast<T*>(buffer_->getDevicePointer()) : nullptr;
    }
    const T* data() const noexcept {
        return buffer_ ? static_cast<const T*>(buffer_->getDevicePointer()) : nullptr;
    }

    /// Get number of elements
    size_t size() const noexcept { return count_; }

    /// Get size in bytes
    size_t size_bytes() const noexcept { return count_ * sizeof(T); }

    /// Check if buffer is empty
    bool empty() const noexcept { return count_ == 0; }

    /// Check if buffer is valid
    explicit operator bool() const noexcept { return buffer_ != nullptr; }

    /// Get device
    Device* device() noexcept { return device_; }
    const Device* device() const noexcept { return device_; }

    /// Get underlying buffer (for direct Queue operations)
    Buffer* buffer() noexcept { return buffer_.get(); }
    const Buffer* buffer() const noexcept { return buffer_.get(); }

    /// Copy from host memory (asynchronous).
    /// @param src Host memory source
    /// @param count Number of elements to copy
    /// @param queue Queue for async operation
    void copy_from_host(const T* src, size_t count, Queue& queue) {
        if (count == 0) return;
        if (count > count_) {
            throw std::out_of_range("Copy count exceeds buffer size");
        }
        if (!buffer_) {
            throw std::runtime_error("Cannot copy to uninitialized buffer");
        }

        // Create temporary pinned host buffer
        auto temp_buffer = device_->createBuffer(count * sizeof(T), MemoryKind::HostPinned);
        if (!temp_buffer) {
            throw std::runtime_error("Failed to allocate temporary host buffer");
        }

        // Copy data to pinned memory
        std::memcpy(temp_buffer->getHostPointer(), src, count * sizeof(T));

        // Async copy from pinned to device
        queue.memcpy(*buffer_, *temp_buffer, count * sizeof(T));
    }

    /// Copy to host memory (synchronous - waits for completion).
    /// @param dst Host memory destination
    /// @param count Number of elements to copy
    /// @param queue Queue for operation
    void copy_to_host(T* dst, size_t count, Queue& queue) const {
        if (count == 0) return;
        if (count > count_) {
            throw std::out_of_range("Copy count exceeds buffer size");
        }
        if (!buffer_) {
            throw std::runtime_error("Cannot copy from uninitialized buffer");
        }

        // Create temporary pinned host buffer
        auto temp_buffer = device_->createBuffer(count * sizeof(T), MemoryKind::HostPinned);
        if (!temp_buffer) {
            throw std::runtime_error("Failed to allocate temporary host buffer");
        }

        // Async copy from device to pinned
        queue.memcpy(*temp_buffer, *buffer_, count * sizeof(T));

        // Wait for completion
        queue.synchronize();

        // Copy from pinned to destination
        std::memcpy(dst, temp_buffer->getHostPointer(), count * sizeof(T));
    }

    /// Resize buffer (allocates new memory, does NOT preserve contents).
    /// @param new_count New number of elements
    void resize(size_t new_count) {
        if (new_count == count_) return;

        count_ = new_count;
        if (count_ > 0 && device_) {
            buffer_ = device_->createBuffer(size_bytes(), MemoryKind::Device);
            if (!buffer_) {
                count_ = 0;
                throw std::runtime_error("Failed to resize device buffer");
            }
        } else {
            buffer_.reset();
        }
    }

    /// Resize buffer preserving contents (expensive - copies data).
    /// @param new_count New number of elements
    /// @param queue Queue for copy operation
    void resize_preserve(size_t new_count, Queue& queue) {
        if (new_count == count_) return;
        if (new_count == 0) {
            buffer_.reset();
            count_ = 0;
            return;
        }

        auto new_buffer = device_->createBuffer(new_count * sizeof(T), MemoryKind::Device);
        if (!new_buffer) {
            throw std::runtime_error("Failed to resize device buffer");
        }

        // Copy old data if any
        if (buffer_ && count_ > 0) {
            size_t copy_count = std::min(count_, new_count);
            queue.memcpy(*new_buffer, *buffer_, copy_count * sizeof(T));
        }

        buffer_ = std::move(new_buffer);
        count_ = new_count;
    }

    /// Release ownership and return buffer.
    /// Caller is responsible for the buffer lifetime.
    std::unique_ptr<Buffer> release() noexcept {
        count_ = 0;
        return std::move(buffer_);
    }

private:
    Device* device_ = nullptr;
    std::unique_ptr<Buffer> buffer_;
    size_t count_ = 0;
};

/// Single device value (useful for reductions).
///
/// Wraps a single value in device memory for GPU operations
/// that produce scalar results.
template<typename T>
class DeviceScalar {
public:
    static_assert(std::is_trivially_copyable<T>::value,
                  "DeviceScalar only supports trivially copyable types");

    /// Construct uninitialized scalar.
    explicit DeviceScalar(Device& device, MemoryKind kind = MemoryKind::Device)
        : buffer_(device, 1, kind)
    {
    }

    /// Construct with initial value (synchronous).
    DeviceScalar(Device& device, T initial_value, Queue& queue, MemoryKind kind = MemoryKind::Device)
        : buffer_(device, 1, kind)
    {
        set(initial_value, queue);
    }

    /// Get pointer to device value
    T* data() noexcept { return buffer_.data(); }
    const T* data() const noexcept { return buffer_.data(); }

    /// Get underlying buffer
    Buffer* buffer() noexcept { return buffer_.buffer(); }
    const Buffer* buffer() const noexcept { return buffer_.buffer(); }

    /// Get value (synchronous read from device)
    T get(Queue& queue) const {
        T result;
        buffer_.copy_to_host(&result, 1, queue);
        return result;
    }

    /// Set value (asynchronous write to device)
    void set(T value, Queue& queue) {
        buffer_.copy_from_host(&value, 1, queue);
    }

    /// Set value and synchronize
    void set_sync(T value, Queue& queue) {
        set(value, queue);
        queue.synchronize();
    }

private:
    DeviceBuffer<T> buffer_;
};

} // namespace GpuRuntime
