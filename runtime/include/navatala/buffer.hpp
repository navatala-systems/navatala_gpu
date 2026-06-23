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

// Navatala C++ Wrapper Layer - Device Buffer
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"
#include "error.hpp"
#include "types.hpp"
#include "queue.hpp"
#include "resource.hpp"

#include <span>
#include <vector>
#include <numeric>
#include <cstring>
#include <fstream>

/// @file navatala/buffer.hpp
/// @brief RAII wrapper for device-resident memory buffers.
///
/// This header provides:
/// - buffer: GPU memory buffer with shape and type tracking
/// - Host-device copy operations via std::span
/// - Factory methods for creation and wrapping existing memory

// Forward declare FFI types
struct NavatalaGpuBuffer;

namespace navatala {

// ============================================================================
// Buffer Class
// ============================================================================

/// @brief RAII wrapper for device-resident memory buffers.
///
/// A buffer represents a contiguous block of GPU memory with associated
/// shape and data type metadata. It supports:
/// - Creation from host data
/// - Wrapping existing device pointers
/// - Host-device and device-host copies
/// - Reshape operations (metadata only)
///
/// Example:
/// @code
///   auto res = resources::create().value();
///
///   // Create buffer from host data
///   std::vector<float> host_data(1024, 1.0f);
///   auto buf = buffer::create(res, std::span(host_data), {32, 32}, data_type::float32);
///
///   // Copy back to host
///   std::vector<float> result(1024);
///   buf.value().copy_to_host(std::as_writable_bytes(std::span(result)), res.default_queue());
///   res.default_queue().sync();
/// @endcode
///
/// @thread_safety Buffer operations that involve the same queue must be serialized.
///                Different buffers can be used from different threads with different queues.
class buffer {
public:
    // -------------------- Factory Methods --------------------

    /// Creates a buffer from host data.
    /// @tparam T The element type of the host data.
    /// @param res The GPU resources.
    /// @param host_data A span of host data to copy to the device.
    /// @param shape The logical shape of the data.
    /// @param dtype The data type of elements.
    /// @param q The queue to use for the host-to-device copy.
    /// @return An expected<buffer> containing the new buffer or an error.
    template<typename T>
    [[nodiscard]] static result<buffer> create(
        resources& res,
        std::span<const T> host_data,
        const std::vector<size_t>& shape,
        data_type dtype,
        const queue& q);

    /// Creates a buffer from host data using the default queue.
    template<typename T>
    [[nodiscard]] static result<buffer> create(
        resources& res,
        std::span<const T> host_data,
        const std::vector<size_t>& shape,
        data_type dtype);

    /// Creates an uninitialized buffer with the specified shape and type.
    /// @param res The GPU resources.
    /// @param shape The logical shape of the buffer.
    /// @param dtype The data type of elements.
    /// @return An expected<buffer> containing the new buffer or an error.
    [[nodiscard]] static result<buffer> create(
        resources& res,
        const std::vector<size_t>& shape,
        data_type dtype);

    /// Creates an uninitialized buffer with specified byte size.
    /// @param res The GPU resources.
    /// @param bytes The size in bytes.
    /// @return An expected<buffer> containing the new buffer or an error.
    [[nodiscard]] static result<buffer> create_bytes(
        resources& res,
        size_t bytes);

    /// Wraps an existing device pointer (non-owning).
    /// @param res The GPU resources.
    /// @param device_ptr A pointer to device memory.
    /// @param shape The logical shape of the data.
    /// @param dtype The data type of elements.
    /// @return An expected<buffer> containing the wrapper or an error.
    /// @warning The caller is responsible for the lifetime of device_ptr.
    [[nodiscard]] static result<buffer> wrap_device_ptr(
        resources& res,
        void* device_ptr,
        const std::vector<size_t>& shape,
        data_type dtype);

    /// Wraps an existing device pointer with ownership transfer.
    ///
    /// The buffer will free the device memory when destroyed. This assumes
    /// the pointer was allocated via the same allocator (context) as the
    /// resources instance.
    ///
    /// @param res The GPU resources.
    /// @param device_ptr A pointer to device memory to take ownership of.
    /// @param shape The logical shape of the data.
    /// @param dtype The data type of elements.
    /// @return An expected<buffer> containing the wrapper or an error.
    ///
    /// @warning The pointer must have been allocated via the same GPU context
    ///          as the resources object. Using a pointer from a different
    ///          context or allocator will result in undefined behavior.
    ///
    /// @note Future versions may add a custom deleter parameter for pointers
    ///       allocated via different mechanisms.
    [[nodiscard]] static result<buffer> wrap_device_ptr_owning(
        resources& res,
        void* device_ptr,
        const std::vector<size_t>& shape,
        data_type dtype);

    /// Wraps an existing FFI buffer handle.
    /// @param res The GPU resources.
    /// @param handle The FFI buffer handle.
    /// @param shape The logical shape of the data.
    /// @param dtype The data type of elements.
    /// @param owns_memory Whether to take ownership of the handle.
    /// @return An expected<buffer> containing the wrapper or an error.
    [[nodiscard]] static result<buffer> wrap_handle(
        resources& res,
        NavatalaGpuBuffer* handle,
        const std::vector<size_t>& shape,
        data_type dtype,
        bool owns_memory = false);

    /// Creates a buffer from a binary file.
    /// @param res The GPU resources.
    /// @param path Path to the binary file.
    /// @param shape The logical shape of the data.
    /// @param dtype The data type of elements.
    /// @param q The queue to use for the copy.
    /// @return An expected<buffer> containing the new buffer or an error.
    [[nodiscard]] static result<buffer> from_file(
        resources& res,
        const std::string& path,
        const std::vector<size_t>& shape,
        data_type dtype,
        const queue& q);

    /// Creates a buffer from a binary file using the default queue.
    [[nodiscard]] static result<buffer> from_file(
        resources& res,
        const std::string& path,
        const std::vector<size_t>& shape,
        data_type dtype);

    // -------------------- Constructors/Destructor --------------------

    /// Default constructor creates an invalid buffer.
    buffer() noexcept;

    /// Destructor. Releases device memory if owned.
    ~buffer() noexcept;

    // Non-copyable
    buffer(const buffer&) = delete;
    buffer& operator=(const buffer&) = delete;

    // Movable
    buffer(buffer&& other) noexcept;
    buffer& operator=(buffer&& other) noexcept;

    // -------------------- Accessors --------------------

    /// Returns a raw pointer to the device memory.
    [[nodiscard]] void* data() const noexcept { return device_ptr_; }

    /// Returns a typed pointer to the device memory.
    template<typename T>
    [[nodiscard]] T* data() const noexcept {
        return static_cast<T*>(device_ptr_);
    }

    /// Returns the logical shape of the buffer.
    [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return shape_; }

    /// Returns the data type of elements.
    [[nodiscard]] data_type dtype() const noexcept { return dtype_; }

    /// Returns the total size in bytes.
    [[nodiscard]] size_t bytes() const noexcept { return total_bytes_; }

    /// Returns the number of elements.
    [[nodiscard]] size_t num_elements() const noexcept {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(),
                               size_t(1), std::multiplies<size_t>());
    }

    /// Returns the number of dimensions.
    [[nodiscard]] size_t ndim() const noexcept { return shape_.size(); }

    /// Returns true if the buffer is valid.
    [[nodiscard]] bool is_valid() const noexcept { return handle_ != nullptr; }

    /// Explicit bool conversion.
    explicit operator bool() const noexcept { return is_valid(); }

    /// Returns whether this buffer owns its memory.
    [[nodiscard]] bool owns_memory() const noexcept { return owns_memory_; }

    /// Returns the raw FFI handle. Internal use only.
    [[nodiscard]] NavatalaGpuBuffer* handle() const noexcept { return handle_; }

    // -------------------- Copy Operations --------------------

    /// Copies data from host memory to this buffer.
    /// @param host_data A span of bytes to copy.
    /// @param q The queue to use for the copy.
    void copy_from_host(std::span<const std::byte> host_data, const queue& q);

    /// Copies data from host memory using the default queue.
    void copy_from_host(std::span<const std::byte> host_data, resources& res);

    /// Copies typed data from host memory.
    template<typename T>
    void copy_from_host(std::span<const T> host_data, const queue& q) {
        copy_from_host(std::as_bytes(host_data), q);
    }

    /// Copies data from this buffer to host memory.
    /// @param host_data A span of bytes to write to.
    /// @param q The queue to use for the copy.
    void copy_to_host(std::span<std::byte> host_data, const queue& q) const;

    /// Copies data to host using the default queue.
    void copy_to_host(std::span<std::byte> host_data, resources& res) const;

    /// Copies typed data to host memory.
    template<typename T>
    void copy_to_host(std::span<T> host_data, const queue& q) const {
        copy_to_host(std::as_writable_bytes(host_data), q);
    }

    /// Copies data from another buffer to this buffer.
    /// @param src The source buffer.
    /// @param q The queue to use for the copy.
    void copy_from(const buffer& src, const queue& q);

    /// Returns the data as a host vector (synchronous).
    /// @tparam T The element type.
    /// @param res The resources for synchronization.
    /// @return A vector containing the buffer data.
    template<typename T>
    [[nodiscard]] std::vector<T> to_host(resources& res) const {
        std::vector<T> result(num_elements());
        copy_to_host(std::span<T>(result), res.default_queue());
        res.default_queue().sync();
        return result;
    }

    // -------------------- File Operations --------------------

    /// Writes the buffer content to a binary file.
    /// @param res The GPU resources.
    /// @param path Path to the output file.
    /// @param q The queue to use for the device-to-host copy.
    void to_file(resources& res, const std::string& path, const queue& q) const;

    /// Writes to file using the default queue.
    void to_file(resources& res, const std::string& path) const;

    // -------------------- Shape Operations --------------------

    /// Returns a new buffer with a different shape (same underlying memory).
    /// @param new_shape The new logical shape.
    /// @return An expected<buffer> with the new shape or an error.
    /// @note The total number of elements must remain the same.
    [[nodiscard]] result<buffer> reshape(const std::vector<size_t>& new_shape) const;

    /// Returns a view of a slice of the buffer.
    /// @param offset Byte offset from the start.
    /// @param size Number of bytes in the slice.
    /// @return An expected<buffer> representing the slice or an error.
    /// @warning The returned buffer does not own its memory.
    [[nodiscard]] result<buffer> slice_bytes(size_t offset, size_t size) const;

    /// Fills the buffer with zeros.
    /// @param q The queue to use.
    void fill_zero(const queue& q);

    /// Swap two buffer objects.
    friend void swap(buffer& a, buffer& b) noexcept;

private:
    // Private constructor
    buffer(NavatalaGpuBuffer* handle, void* device_ptr, resources* res,
           std::vector<size_t> shape, data_type dtype, size_t bytes, bool owns);

    NavatalaGpuBuffer* handle_;
    void* device_ptr_;
    resources* parent_resources_;
    std::vector<size_t> shape_;
    data_type dtype_;
    size_t total_bytes_;
    bool owns_memory_;
};

// ============================================================================
// Inline Implementations
// ============================================================================

inline buffer::buffer() noexcept
    : handle_(nullptr)
    , device_ptr_(nullptr)
    , parent_resources_(nullptr)
    , shape_()
    , dtype_(data_type::float32)
    , total_bytes_(0)
    , owns_memory_(false)
{}

inline buffer::buffer(NavatalaGpuBuffer* handle, void* device_ptr, resources* res,
                      std::vector<size_t> shape, data_type dtype, size_t bytes, bool owns)
    : handle_(handle)
    , device_ptr_(device_ptr)
    , parent_resources_(res)
    , shape_(std::move(shape))
    , dtype_(dtype)
    , total_bytes_(bytes)
    , owns_memory_(owns)
{}

inline buffer::buffer(buffer&& other) noexcept
    : handle_(other.handle_)
    , device_ptr_(other.device_ptr_)
    , parent_resources_(other.parent_resources_)
    , shape_(std::move(other.shape_))
    , dtype_(other.dtype_)
    , total_bytes_(other.total_bytes_)
    , owns_memory_(other.owns_memory_)
{
    other.handle_ = nullptr;
    other.device_ptr_ = nullptr;
    other.parent_resources_ = nullptr;
    other.total_bytes_ = 0;
    other.owns_memory_ = false;
}

inline buffer& buffer::operator=(buffer&& other) noexcept {
    if (this != &other) {
        buffer tmp(std::move(other));
        swap(*this, tmp);
    }
    return *this;
}

inline void swap(buffer& a, buffer& b) noexcept {
    using std::swap;
    swap(a.handle_, b.handle_);
    swap(a.device_ptr_, b.device_ptr_);
    swap(a.parent_resources_, b.parent_resources_);
    swap(a.shape_, b.shape_);
    swap(a.dtype_, b.dtype_);
    swap(a.total_bytes_, b.total_bytes_);
    swap(a.owns_memory_, b.owns_memory_);
}

// Template implementations
template<typename T>
result<buffer> buffer::create(
    resources& res,
    std::span<const T> host_data,
    const std::vector<size_t>& shape,
    data_type dtype,
    const queue& q)
{
    // Calculate expected size
    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }

    if (host_data.size() < num_elements) {
        return result<buffer>(error_code::invalid_param,
                              "host_data size smaller than shape requires");
    }

    size_t element_size = sizeof_data_type(dtype);
    size_t total_bytes = num_elements * element_size;

    // Create uninitialized buffer
    auto buf_result = create(res, shape, dtype);
    if (!buf_result.has_value()) {
        return buf_result;
    }

    // Copy host data
    buf_result.value().copy_from_host(
        std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(host_data.data()),
            total_bytes),
        q);

    return buf_result;
}

template<typename T>
result<buffer> buffer::create(
    resources& res,
    std::span<const T> host_data,
    const std::vector<size_t>& shape,
    data_type dtype)
{
    return create(res, host_data, shape, dtype, res.default_queue());
}

} // namespace navatala
