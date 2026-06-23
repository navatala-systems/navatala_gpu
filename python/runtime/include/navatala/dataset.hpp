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

// Navatala C++ Wrapper Layer - Dataset
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"
#include "error.hpp"
#include "types.hpp"
#include "queue.hpp"
#include "resource.hpp"
#include "buffer.hpp"
#include "navatala_ffi.h"

#include <span>
#include <vector>
#include <cstring>

/// @file navatala/dataset.hpp
/// @brief RAII wrapper for structured vector datasets.
///
/// This header provides:
/// - dataset: GPU-resident vector collection for search operations
/// - Factory methods for creation from host data or existing buffers
/// - Row/column access utilities
///
/// Datasets are 2D arrays optimized for vector search:
/// - First dimension: number of vectors (n_vectors)
/// - Second dimension: vector dimensionality (dims)

// Forward declare FFI types
struct NavatalaGpuBuffer;

namespace navatala {

// ============================================================================
// Dataset Class
// ============================================================================

/// @brief RAII wrapper for structured vector datasets.
///
/// A dataset represents a collection of vectors for search operations.
/// It maintains metadata about vector count, dimensionality, and data type.
///
/// Example:
/// @code
///   auto res = resources::create().value();
///
///   // Create dataset from host vectors
///   std::vector<float> vectors(1000 * 128);  // 1000 vectors of 128 dims
///   auto ds = dataset::create(res, std::span(vectors), 1000, 128, data_type::float32);
///
///   // Access properties
///   std::cout << "Vectors: " << ds.value().n_vectors() << std::endl;
///   std::cout << "Dims: " << ds.value().dims() << std::endl;
/// @endcode
///
/// @thread_safety Dataset operations that involve the same queue must be serialized.
///                Different datasets can be used from different threads.
class dataset {
public:
    // -------------------- Factory Methods --------------------

    /// Creates a dataset from host data.
    /// @tparam T The element type of the host data.
    /// @param res The GPU resources.
    /// @param host_data A span of host data to copy to the device.
    /// @param n_vectors Number of vectors in the dataset.
    /// @param dims Dimensionality of each vector.
    /// @param dtype The data type of elements.
    /// @param q The queue to use for the host-to-device copy.
    /// @return An expected<dataset> containing the new dataset or an error.
    template<typename T>
    [[nodiscard]] static result<dataset> create(
        resources& res,
        std::span<const T> host_data,
        size_t n_vectors,
        size_t dims,
        data_type dtype,
        const queue& q);

    /// Creates a dataset from host data using the default queue.
    template<typename T>
    [[nodiscard]] static result<dataset> create(
        resources& res,
        std::span<const T> host_data,
        size_t n_vectors,
        size_t dims,
        data_type dtype);

    /// Creates an uninitialized dataset with the specified dimensions.
    /// @param res The GPU resources.
    /// @param n_vectors Number of vectors.
    /// @param dims Dimensionality of each vector.
    /// @param dtype The data type of elements.
    /// @return An expected<dataset> containing the new dataset or an error.
    [[nodiscard]] static result<dataset> create(
        resources& res,
        size_t n_vectors,
        size_t dims,
        data_type dtype);

    /// Wraps an existing buffer as a dataset (non-owning).
    /// @param buf The buffer to wrap.
    /// @param n_vectors Number of vectors.
    /// @param dims Dimensionality of each vector.
    /// @return An expected<dataset> containing the wrapper or an error.
    /// @warning The caller is responsible for the lifetime of the buffer.
    [[nodiscard]] static result<dataset> wrap_buffer(
        buffer& buf,
        size_t n_vectors,
        size_t dims);

    /// Wraps an existing buffer as a dataset (owning).
    /// @param buf The buffer to take ownership of (moved).
    /// @param n_vectors Number of vectors.
    /// @param dims Dimensionality of each vector.
    /// @return An expected<dataset> containing the wrapper or an error.
    [[nodiscard]] static result<dataset> wrap_buffer(
        buffer&& buf,
        size_t n_vectors,
        size_t dims);

    // -------------------- Constructors/Destructor --------------------

    /// Default constructor creates an invalid dataset.
    dataset() noexcept;

    /// Destructor. Releases resources if owned.
    ~dataset() noexcept;

    // Non-copyable
    dataset(const dataset&) = delete;
    dataset& operator=(const dataset&) = delete;

    // Movable
    dataset(dataset&& other) noexcept;
    dataset& operator=(dataset&& other) noexcept;

    // -------------------- Accessors --------------------

    /// Returns the number of vectors in the dataset.
    [[nodiscard]] size_t n_vectors() const noexcept { return n_vectors_; }

    /// Returns the dimensionality of each vector.
    [[nodiscard]] size_t dims() const noexcept { return dims_; }

    /// Returns the total number of elements (n_vectors * dims).
    [[nodiscard]] size_t size() const noexcept { return n_vectors_ * dims_; }

    /// Returns the size of one vector in bytes.
    [[nodiscard]] size_t stride_bytes() const noexcept {
        return dims_ * sizeof_data_type(dtype_);
    }

    /// Returns the total size in bytes.
    [[nodiscard]] size_t bytes() const noexcept {
        return n_vectors_ * stride_bytes();
    }

    /// Returns the data type of elements.
    [[nodiscard]] data_type dtype() const noexcept { return dtype_; }

    /// Returns the base pointer to device memory (without offset).
    /// @note For sliced datasets, use device_ptr() to get the offset-adjusted pointer.
    [[nodiscard]] void* base_ptr() const noexcept {
        return buffer_.is_valid() ? buffer_.data() : external_ptr_;
    }

    /// Returns the byte offset into the base pointer for this view.
    [[nodiscard]] size_t offset_bytes() const noexcept { return offset_bytes_; }

    /// Returns a raw pointer to the device memory (offset-adjusted for sliced views).
    [[nodiscard]] void* device_ptr() const noexcept {
        void* base = base_ptr();
        if (base == nullptr || offset_bytes_ == 0) {
            return base;
        }
        return static_cast<char*>(base) + offset_bytes_;
    }

    /// Returns a typed pointer to the device memory.
    template<typename T>
    [[nodiscard]] T* device_ptr() const noexcept {
        return static_cast<T*>(device_ptr());
    }

    /// Returns true if the dataset is valid.
    [[nodiscard]] bool is_valid() const noexcept {
        return buffer_.is_valid() || external_ptr_ != nullptr;
    }

    /// Explicit bool conversion.
    explicit operator bool() const noexcept { return is_valid(); }

    /// Returns whether this dataset owns its underlying buffer.
    [[nodiscard]] bool owns_buffer() const noexcept { return owns_buffer_; }

    /// Returns the underlying buffer (if owned).
    [[nodiscard]] const buffer& get_buffer() const noexcept { return buffer_; }

    // -------------------- Copy Operations --------------------

    /// Copies data from host memory to this dataset.
    /// @param host_data A span of bytes to copy.
    /// @param q The queue to use for the copy.
    void copy_from_host(std::span<const std::byte> host_data, const queue& q);

    /// Copies typed data from host memory.
    template<typename T>
    void copy_from_host(std::span<const T> host_data, const queue& q) {
        copy_from_host(std::as_bytes(host_data), q);
    }

    /// Copies data from this dataset to host memory.
    /// @param host_data A span of bytes to write to.
    /// @param q The queue to use for the copy.
    void copy_to_host(std::span<std::byte> host_data, const queue& q) const;

    /// Copies typed data to host memory.
    template<typename T>
    void copy_to_host(std::span<T> host_data, const queue& q) const {
        copy_to_host(std::as_writable_bytes(host_data), q);
    }

    /// Returns the data as a host vector (synchronous).
    /// @tparam T The element type.
    /// @param res The resources for synchronization.
    /// @return A vector containing the dataset data.
    template<typename T>
    [[nodiscard]] std::vector<T> to_host(resources& res) const {
        std::vector<T> result(size());
        copy_to_host(std::span<T>(result), res.default_queue());
        res.default_queue().sync();
        return result;
    }

    // -------------------- Row Access --------------------

    /// Returns a view of a single vector (row) in the dataset.
    /// @param index The vector index.
    /// @return An expected<dataset> representing the single vector.
    /// @warning The returned dataset does not own its memory.
    /// @note Uses offset-based slicing internally. The returned view
    ///       tracks its offset from the base pointer, avoiding assumptions
    ///       about linear device pointer arithmetic.
    [[nodiscard]] result<dataset> row(size_t index) const;

    /// Returns a view of a range of vectors.
    /// @param start The starting vector index.
    /// @param count Number of vectors to include.
    /// @return An expected<dataset> representing the slice.
    /// @warning The returned dataset does not own its memory.
    /// @note Uses offset-based slicing internally. See row() for details.
    [[nodiscard]] result<dataset> rows(size_t start, size_t count) const;

private:
    // Private constructor for internal use
    dataset(buffer buf, size_t n_vectors, size_t dims, data_type dtype, bool owns);

    // Private constructor for non-owning view
    dataset(void* external_ptr, size_t n_vectors, size_t dims, data_type dtype);

    // Private constructor for offset-based view (slicing) with parent buffer tracking
    dataset(void* base_ptr, size_t offset_bytes, size_t n_vectors, size_t dims, data_type dtype, const buffer* parent);

    buffer buffer_;           // Owned buffer (may be invalid if external_ptr_ is used)
    void* external_ptr_;      // External pointer (non-owning case)
    size_t offset_bytes_;     // Byte offset into buffer/external_ptr for sliced views
    size_t n_vectors_;
    size_t dims_;
    data_type dtype_;
    bool owns_buffer_;

    // For sliced views, track parent buffer for FFI operations
    const buffer* parent_buffer_ = nullptr;
};

// ============================================================================
// Inline Implementations
// ============================================================================

inline dataset::dataset() noexcept
    : buffer_()
    , external_ptr_(nullptr)
    , offset_bytes_(0)
    , n_vectors_(0)
    , dims_(0)
    , dtype_(data_type::float32)
    , owns_buffer_(false)
{}

inline dataset::dataset(buffer buf, size_t n_vectors, size_t dims, data_type dtype, bool owns)
    : buffer_(std::move(buf))
    , external_ptr_(nullptr)
    , offset_bytes_(0)
    , n_vectors_(n_vectors)
    , dims_(dims)
    , dtype_(dtype)
    , owns_buffer_(owns)
{}

inline dataset::dataset(void* external_ptr, size_t n_vectors, size_t dims, data_type dtype)
    : buffer_()
    , external_ptr_(external_ptr)
    , offset_bytes_(0)
    , n_vectors_(n_vectors)
    , dims_(dims)
    , dtype_(dtype)
    , owns_buffer_(false)
{}

inline dataset::dataset(void* base_ptr, size_t offset_bytes, size_t n_vectors,
                        size_t dims, data_type dtype, const buffer* parent)
    : buffer_()
    , external_ptr_(base_ptr)
    , offset_bytes_(offset_bytes)
    , n_vectors_(n_vectors)
    , dims_(dims)
    , dtype_(dtype)
    , owns_buffer_(false)
    , parent_buffer_(parent)
{}

inline dataset::~dataset() noexcept {
    // Buffer's destructor handles cleanup if owned
}

inline dataset::dataset(dataset&& other) noexcept
    : buffer_(std::move(other.buffer_))
    , external_ptr_(other.external_ptr_)
    , offset_bytes_(other.offset_bytes_)
    , n_vectors_(other.n_vectors_)
    , dims_(other.dims_)
    , dtype_(other.dtype_)
    , owns_buffer_(other.owns_buffer_)
    , parent_buffer_(other.parent_buffer_)
{
    other.external_ptr_ = nullptr;
    other.offset_bytes_ = 0;
    other.n_vectors_ = 0;
    other.dims_ = 0;
    other.owns_buffer_ = false;
    other.parent_buffer_ = nullptr;
}

inline dataset& dataset::operator=(dataset&& other) noexcept {
    if (this != &other) {
        buffer_ = std::move(other.buffer_);
        external_ptr_ = other.external_ptr_;
        offset_bytes_ = other.offset_bytes_;
        n_vectors_ = other.n_vectors_;
        dims_ = other.dims_;
        dtype_ = other.dtype_;
        owns_buffer_ = other.owns_buffer_;
        parent_buffer_ = other.parent_buffer_;

        other.external_ptr_ = nullptr;
        other.offset_bytes_ = 0;
        other.n_vectors_ = 0;
        other.dims_ = 0;
        other.owns_buffer_ = false;
        other.parent_buffer_ = nullptr;
    }
    return *this;
}

// Template factory implementations
template<typename T>
result<dataset> dataset::create(
    resources& res,
    std::span<const T> host_data,
    size_t n_vectors,
    size_t dims,
    data_type dtype,
    const queue& q)
{
    size_t expected_elements = n_vectors * dims;
    if (host_data.size() < expected_elements) {
        return result<dataset>(error_code::invalid_param,
                               "host_data size smaller than n_vectors * dims");
    }

    // Create underlying buffer
    auto buf_result = buffer::create(res, host_data, {n_vectors, dims}, dtype, q);
    if (!buf_result.has_value()) {
        return result<dataset>(buf_result.error(), "failed to create dataset buffer");
    }

    return result<dataset>(dataset(std::move(buf_result.value()), n_vectors, dims, dtype, true));
}

template<typename T>
result<dataset> dataset::create(
    resources& res,
    std::span<const T> host_data,
    size_t n_vectors,
    size_t dims,
    data_type dtype)
{
    return create(res, host_data, n_vectors, dims, dtype, res.default_queue());
}

inline result<dataset> dataset::create(
    resources& res,
    size_t n_vectors,
    size_t dims,
    data_type dtype)
{
    // Create underlying buffer
    auto buf_result = buffer::create(res, {n_vectors, dims}, dtype);
    if (!buf_result.has_value()) {
        return result<dataset>(buf_result.error(), "failed to create dataset buffer");
    }

    return result<dataset>(dataset(std::move(buf_result.value()), n_vectors, dims, dtype, true));
}

inline result<dataset> dataset::wrap_buffer(
    buffer& buf,
    size_t n_vectors,
    size_t dims)
{
    if (!buf.is_valid()) {
        return result<dataset>(error_code::invalid_handle, "buffer is invalid");
    }

    size_t required_bytes = n_vectors * dims * sizeof_data_type(buf.dtype());
    if (buf.bytes() < required_bytes) {
        return result<dataset>(error_code::invalid_param,
                               "buffer too small for specified dimensions");
    }

    // Non-owning view - use external pointer
    return result<dataset>(dataset(buf.data(), n_vectors, dims, buf.dtype()));
}

inline result<dataset> dataset::wrap_buffer(
    buffer&& buf,
    size_t n_vectors,
    size_t dims)
{
    if (!buf.is_valid()) {
        return result<dataset>(error_code::invalid_handle, "buffer is invalid");
    }

    size_t required_bytes = n_vectors * dims * sizeof_data_type(buf.dtype());
    if (buf.bytes() < required_bytes) {
        return result<dataset>(error_code::invalid_param,
                               "buffer too small for specified dimensions");
    }

    data_type dtype = buf.dtype();
    return result<dataset>(dataset(std::move(buf), n_vectors, dims, dtype, true));
}

inline void dataset::copy_from_host(std::span<const std::byte> host_data, const queue& q) {
    if (buffer_.is_valid()) {
        // Owning buffer - use buffer's copy method
        buffer_.copy_from_host(host_data, q);
    } else if (parent_buffer_ && parent_buffer_->is_valid()) {
        // Sliced view - use offset-aware FFI
        size_t copy_bytes = std::min(host_data.size(), bytes());
        NavatalaErrorCode err = navatala_gpu_copy_h2d_offset(
            parent_buffer_->handle(),
            offset_bytes_,
            host_data.data(),
            copy_bytes,
            q.handle());
        if (err != NAVATALA_SUCCESS) {
            NAVATALA_THROW_OR_ABORT(translate_error(err), "dataset::copy_from_host (slice) failed");
        }
    }
    // For raw external pointers without parent, still a no-op (unchanged behavior)
}

inline void dataset::copy_to_host(std::span<std::byte> host_data, const queue& q) const {
    if (buffer_.is_valid()) {
        buffer_.copy_to_host(host_data, q);
    } else if (parent_buffer_ && parent_buffer_->is_valid()) {
        // Sliced view - use offset-aware FFI
        size_t copy_bytes = std::min(host_data.size(), bytes());
        NavatalaErrorCode err = navatala_gpu_copy_d2h_offset(
            parent_buffer_->handle(),
            offset_bytes_,
            host_data.data(),
            copy_bytes,
            q.handle());
        if (err != NAVATALA_SUCCESS) {
            NAVATALA_THROW_OR_ABORT(translate_error(err), "dataset::copy_to_host (slice) failed");
        }
    }
}

inline result<dataset> dataset::row(size_t index) const {
    if (index >= n_vectors_) {
        return result<dataset>(error_code::out_of_bounds,
                               "row index out of bounds");
    }

    // Use offset-based slicing to avoid pointer arithmetic assumptions.
    // The new_offset is relative to the base pointer, accumulating any
    // existing offset from this dataset (if it's already a slice).
    size_t new_offset = offset_bytes_ + index * stride_bytes();
    const buffer* parent = buffer_.is_valid() ? &buffer_ : parent_buffer_;
    return result<dataset>(dataset(base_ptr(), new_offset, 1, dims_, dtype_, parent));
}

inline result<dataset> dataset::rows(size_t start, size_t count) const {
    if (start + count > n_vectors_) {
        return result<dataset>(error_code::out_of_bounds,
                               "row range out of bounds");
    }

    // Use offset-based slicing to avoid pointer arithmetic assumptions.
    size_t new_offset = offset_bytes_ + start * stride_bytes();
    const buffer* parent = buffer_.is_valid() ? &buffer_ : parent_buffer_;
    return result<dataset>(dataset(base_ptr(), new_offset, count, dims_, dtype_, parent));
}

} // namespace navatala
