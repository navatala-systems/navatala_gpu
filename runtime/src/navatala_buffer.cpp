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

// Navatala C++ Wrapper Layer - Buffer Implementation
// Part of the Navatala GPU Runtime Library
#include "../include/navatala/buffer.hpp"
#include "../include/navatala/navatala_ffi.h"

#include <fstream>
#include <cstring>

namespace navatala {

// ============================================================================
// Destructor
// ============================================================================

buffer::~buffer() noexcept {
    if (handle_ && owns_memory_) {
        navatala_gpu_destroy_buffer(handle_);
    }
    handle_ = nullptr;
    device_ptr_ = nullptr;
}

// ============================================================================
// Factory Methods - Non-template
// ============================================================================

result<buffer> buffer::create(
    resources& res,
    const std::vector<size_t>& shape,
    data_type dtype)
{
    if (shape.empty()) {
        return result<buffer>(error_code::invalid_param, "shape cannot be empty");
    }

    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }

    size_t element_size = sizeof_data_type(dtype);
    size_t total_bytes = num_elements * element_size;

    NavatalaGpuBuffer* handle = nullptr;
    NavatalaErrorCode err = navatala_gpu_create_buffer(
        res.handle(),
        total_bytes,
        &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "failed to create buffer");
    }

    void* device_ptr = navatala_gpu_buffer_ptr(handle);

    return result<buffer>(buffer(handle, device_ptr, &res, shape, dtype, total_bytes, true));
}

result<buffer> buffer::create_bytes(
    resources& res,
    size_t bytes)
{
    NavatalaGpuBuffer* handle = nullptr;
    NavatalaErrorCode err = navatala_gpu_create_buffer(
        res.handle(),
        bytes,
        &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "failed to create buffer");
    }

    void* device_ptr = navatala_gpu_buffer_ptr(handle);

    return result<buffer>(buffer(handle, device_ptr, &res, {bytes}, data_type::uint8, bytes, true));
}

result<buffer> buffer::wrap_device_ptr(
    resources& res,
    void* device_ptr,
    const std::vector<size_t>& shape,
    data_type dtype)
{
    if (!device_ptr) {
        return result<buffer>(error_code::invalid_param, "device_ptr is null");
    }

    if (shape.empty()) {
        return result<buffer>(error_code::invalid_param, "shape cannot be empty");
    }

    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }

    size_t element_size = sizeof_data_type(dtype);
    size_t total_bytes = num_elements * element_size;

    NavatalaGpuBuffer* handle = nullptr;
    NavatalaErrorCode err = navatala_gpu_wrap_device_ptr(
        res.handle(),
        device_ptr,
        total_bytes,
        &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "failed to wrap device pointer");
    }

    return result<buffer>(buffer(handle, device_ptr, &res, shape, dtype, total_bytes, false));
}

result<buffer> buffer::wrap_device_ptr_owning(
    resources& res,
    void* device_ptr,
    const std::vector<size_t>& shape,
    data_type dtype)
{
    if (!device_ptr) {
        return result<buffer>(error_code::invalid_param, "device_ptr is null");
    }

    if (shape.empty()) {
        return result<buffer>(error_code::invalid_param, "shape cannot be empty");
    }

    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }

    size_t element_size = sizeof_data_type(dtype);
    size_t total_bytes = num_elements * element_size;

    NavatalaGpuBuffer* handle = nullptr;
    NavatalaErrorCode err = navatala_gpu_wrap_device_ptr(
        res.handle(),
        device_ptr,
        total_bytes,
        &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "failed to wrap device pointer");
    }

    // Owning version - will free memory on destruction
    return result<buffer>(buffer(handle, device_ptr, &res, shape, dtype, total_bytes, true));
}

result<buffer> buffer::wrap_handle(
    resources& res,
    NavatalaGpuBuffer* handle,
    const std::vector<size_t>& shape,
    data_type dtype,
    bool owns_memory)
{
    if (!handle) {
        return result<buffer>(error_code::invalid_handle, "handle is null");
    }

    void* device_ptr = navatala_gpu_buffer_ptr(handle);
    size_t total_bytes = navatala_gpu_buffer_size(handle);

    return result<buffer>(buffer(handle, device_ptr, &res, shape, dtype, total_bytes, owns_memory));
}

result<buffer> buffer::from_file(
    resources& res,
    const std::string& path,
    const std::vector<size_t>& shape,
    data_type dtype,
    const queue& q)
{
    // Read file into host memory
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return result<buffer>(error_code::io_error, "failed to open file");
    }

    size_t file_size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    // Calculate expected size
    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }
    size_t expected_size = num_elements * sizeof_data_type(dtype);

    if (file_size < expected_size) {
        return result<buffer>(error_code::invalid_param,
                              "file size smaller than expected");
    }

    std::vector<char> host_data(expected_size);
    if (!file.read(host_data.data(), static_cast<std::streamsize>(expected_size))) {
        return result<buffer>(error_code::io_error, "failed to read file");
    }

    // Create buffer
    auto buf_result = create(res, shape, dtype);
    if (!buf_result.has_value()) {
        return buf_result;
    }

    // Copy data
    buf_result.value().copy_from_host(
        std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(host_data.data()),
            expected_size),
        q);

    return buf_result;
}

result<buffer> buffer::from_file(
    resources& res,
    const std::string& path,
    const std::vector<size_t>& shape,
    data_type dtype)
{
    return from_file(res, path, shape, dtype, res.default_queue());
}

// ============================================================================
// Copy Operations
// ============================================================================

void buffer::copy_from_host(std::span<const std::byte> host_data, const queue& q) {
    if (!handle_) {
        NAVATALA_THROW_OR_ABORT(error_code::invalid_handle, "buffer is invalid");
    }

    size_t copy_size = std::min(host_data.size(), total_bytes_);

    NavatalaErrorCode err = navatala_gpu_copy_h2d(
        handle_,
        host_data.data(),
        copy_size,
        q.handle());

    if (err != NAVATALA_SUCCESS) {
        NAVATALA_THROW_OR_ABORT(translate_error(err), "H2D copy failed");
    }
}

void buffer::copy_from_host(std::span<const std::byte> host_data, resources& res) {
    copy_from_host(host_data, res.default_queue());
}

void buffer::copy_to_host(std::span<std::byte> host_data, const queue& q) const {
    if (!handle_) {
        NAVATALA_THROW_OR_ABORT(error_code::invalid_handle, "buffer is invalid");
    }

    size_t copy_size = std::min(host_data.size(), total_bytes_);

    NavatalaErrorCode err = navatala_gpu_copy_d2h(
        handle_,
        host_data.data(),
        copy_size,
        q.handle());

    if (err != NAVATALA_SUCCESS) {
        NAVATALA_THROW_OR_ABORT(translate_error(err), "D2H copy failed");
    }
}

void buffer::copy_to_host(std::span<std::byte> host_data, resources& res) const {
    copy_to_host(host_data, res.default_queue());
}

void buffer::copy_from(const buffer& src, const queue& q) {
    if (!handle_ || !src.handle_) {
        NAVATALA_THROW_OR_ABORT(error_code::invalid_handle, "buffer is invalid");
    }

    size_t copy_size = std::min(total_bytes_, src.total_bytes_);

    NavatalaErrorCode err = navatala_gpu_copy_d2d(
        handle_,
        src.handle_,
        copy_size,
        q.handle());

    if (err != NAVATALA_SUCCESS) {
        NAVATALA_THROW_OR_ABORT(translate_error(err), "D2D copy failed");
    }
}

// ============================================================================
// File Operations
// ============================================================================

void buffer::to_file(resources& res, const std::string& path, const queue& q) const {
    if (!handle_) {
        NAVATALA_THROW_OR_ABORT(error_code::invalid_handle, "buffer is invalid");
    }

    // Copy to host
    std::vector<std::byte> host_data(total_bytes_);
    copy_to_host(std::span<std::byte>(host_data), q);
    q.sync();

    // Write to file
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        NAVATALA_THROW_OR_ABORT(error_code::io_error, "failed to open file for writing");
    }

    file.write(reinterpret_cast<const char*>(host_data.data()),
               static_cast<std::streamsize>(total_bytes_));

    if (!file) {
        NAVATALA_THROW_OR_ABORT(error_code::io_error, "failed to write file");
    }
}

void buffer::to_file(resources& res, const std::string& path) const {
    to_file(res, path, res.default_queue());
}

// ============================================================================
// Shape Operations
// ============================================================================

result<buffer> buffer::reshape(const std::vector<size_t>& new_shape) const {
    if (!handle_) {
        return result<buffer>(error_code::invalid_handle, "buffer is invalid");
    }

    // Calculate new element count
    size_t new_elements = 1;
    for (size_t dim : new_shape) {
        new_elements *= dim;
    }

    // Must have same total elements
    if (new_elements != num_elements()) {
        return result<buffer>(error_code::invalid_param,
                              "reshape: element count mismatch");
    }

    // Create a non-owning view with new shape
    return result<buffer>(buffer(handle_, device_ptr_, parent_resources_,
                                 new_shape, dtype_, total_bytes_, false));
}

result<buffer> buffer::slice_bytes(size_t offset, size_t size) const {
    if (!handle_) {
        return result<buffer>(error_code::invalid_handle, "buffer is invalid");
    }

    if (offset + size > total_bytes_) {
        return result<buffer>(error_code::out_of_bounds, "slice out of bounds");
    }

    void* slice_ptr = static_cast<char*>(device_ptr_) + offset;

    // Create a non-owning view of the slice
    // We can't easily wrap this with FFI, so we create a "virtual" buffer
    // This is a simplified implementation
    return result<buffer>(buffer(nullptr, slice_ptr, parent_resources_,
                                 {size}, data_type::uint8, size, false));
}

void buffer::fill_zero(const queue& q) {
    if (!handle_) {
        NAVATALA_THROW_OR_ABORT(error_code::invalid_handle, "buffer is invalid");
    }

    NavatalaErrorCode err = navatala_gpu_buffer_fill(handle_, 0, q.handle());

    if (err != NAVATALA_SUCCESS) {
        NAVATALA_THROW_OR_ABORT(translate_error(err), "fill_zero failed");
    }
}

} // namespace navatala
