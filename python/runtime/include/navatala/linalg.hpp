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

// Navatala C++ Wrapper Layer - Portable Linear Algebra Facades
// Part of the Navatala GPU Runtime Library

#pragma once

#include "buffer.hpp"
#include "error.hpp"
#include "navatala_ffi.h"
#include "queue.hpp"
#include "types.hpp"

#include <cstddef>
#include <limits>

/// @file navatala/linalg.hpp
/// @brief Stable C++ facades for portable dense and sparse linear algebra.
///
/// These functions wrap the public C ABI helpers and provide shape/type
/// validation through the C++ buffer API. They are not vendor ABI-compatible
/// cuBLAS/cuSPARSE entry points; they are portable Navatala operations with
/// consistent semantics across supported backends.

namespace navatala::linalg {

enum class matrix_transpose : int {
    none = NAVATALA_MATRIX_OP_NONE,
    transpose = NAVATALA_MATRIX_OP_TRANSPOSE
};

namespace detail {

[[nodiscard]] inline status success_or_error(
    NavatalaErrorCode code,
    const char* operation) noexcept
{
    if (code == NAVATALA_SUCCESS) {
        return status();
    }
    return status(translate_error(static_cast<uint32_t>(code)), operation);
}

[[nodiscard]] inline bool mul_overflows(size_t a, size_t b) noexcept {
    return b != 0 && a > std::numeric_limits<size_t>::max() / b;
}

[[nodiscard]] inline bool add_overflows(size_t a, size_t b) noexcept {
    return a > std::numeric_limits<size_t>::max() - b;
}

[[nodiscard]] inline NavatalaMatrixTranspose to_ffi(matrix_transpose op) noexcept {
    return op == matrix_transpose::transpose
        ? NAVATALA_MATRIX_OP_TRANSPOSE
        : NAVATALA_MATRIX_OP_NONE;
}

[[nodiscard]] inline bool strided_extent_overflows(
    size_t matrix_elements,
    size_t stride,
    size_t batch_count,
    bool allow_broadcast,
    size_t* extent) noexcept
{
    if (!extent) {
        return true;
    }
    if (batch_count == 0) {
        *extent = 0;
        return false;
    }
    if (batch_count == 1) {
        *extent = matrix_elements;
        return false;
    }
    if (matrix_elements == 0) {
        *extent = 0;
        return false;
    }
    if (stride == 0) {
        if (!allow_broadcast) {
            return true;
        }
        *extent = matrix_elements;
        return false;
    }
    if (stride < matrix_elements) {
        return true;
    }
    const size_t batches_before_last = batch_count - 1;
    if (mul_overflows(batches_before_last, stride)) {
        return true;
    }
    const size_t base = batches_before_last * stride;
    if (add_overflows(base, matrix_elements)) {
        return true;
    }
    *extent = base + matrix_elements;
    return false;
}

[[nodiscard]] inline status require_valid_float32(
    const buffer& buf,
    size_t min_elements,
    const char* operation) noexcept
{
    if (!buf.is_valid()) {
        return status(error_code::invalid_handle, operation);
    }
    if (buf.dtype() != data_type::float32) {
        return status(error_code::invalid_param, operation);
    }
    if (buf.num_elements() < min_elements) {
        return status(error_code::invalid_param, operation);
    }
    return status();
}

[[nodiscard]] inline status require_valid_float16(
    const buffer& buf,
    size_t min_elements,
    const char* operation) noexcept
{
    if (!buf.is_valid()) {
        return status(error_code::invalid_handle, operation);
    }
    if (buf.dtype() != data_type::float16) {
        return status(error_code::invalid_param, operation);
    }
    if (buf.num_elements() < min_elements) {
        return status(error_code::invalid_param, operation);
    }
    return status();
}

[[nodiscard]] inline status require_valid_uint32(
    const buffer& buf,
    size_t min_elements,
    const char* operation) noexcept
{
    if (!buf.is_valid()) {
        return status(error_code::invalid_handle, operation);
    }
    if (buf.dtype() != data_type::uint32) {
        return status(error_code::invalid_param, operation);
    }
    if (buf.num_elements() < min_elements) {
        return status(error_code::invalid_param, operation);
    }
    return status();
}

} // namespace detail

/// @brief Compute y = alpha * x + y for single-precision vectors.
[[nodiscard]] inline status axpy(
    const buffer& x,
    buffer& y,
    float alpha,
    size_t element_count,
    const queue& q) noexcept
{
    if (!q.is_valid()) {
        return status(error_code::invalid_handle, "linalg::axpy queue");
    }
    auto sx = detail::require_valid_float32(x, element_count, "linalg::axpy x");
    if (!sx) {
        return sx;
    }
    auto sy = detail::require_valid_float32(y, element_count, "linalg::axpy y");
    if (!sy) {
        return sy;
    }
    return detail::success_or_error(
        navatala_gpu_axpy_f32(x.handle(), y.handle(), alpha, element_count, q.handle()),
        "linalg::axpy");
}

/// @brief Compute C = alpha * A @ B + beta * C for row-major Float32 matrices.
[[nodiscard]] inline status gemm(
    const buffer& a,
    const buffer& b,
    buffer& c,
    size_t m,
    size_t n,
    size_t k,
    float alpha,
    float beta,
    const queue& q) noexcept
{
    if (!q.is_valid()) {
        return status(error_code::invalid_handle, "linalg::gemm queue");
    }
    if (detail::mul_overflows(m, k) ||
        detail::mul_overflows(k, n) ||
        detail::mul_overflows(m, n)) {
        return status(error_code::overflow_error, "linalg::gemm shape");
    }
    auto sa = detail::require_valid_float32(a, m * k, "linalg::gemm A");
    if (!sa) {
        return sa;
    }
    auto sb = detail::require_valid_float32(b, k * n, "linalg::gemm B");
    if (!sb) {
        return sb;
    }
    auto sc = detail::require_valid_float32(c, m * n, "linalg::gemm C");
    if (!sc) {
        return sc;
    }
    return detail::success_or_error(
        navatala_gpu_gemm_f32(
            a.handle(), b.handle(), c.handle(), m, n, k, alpha, beta, q.handle()),
        "linalg::gemm");
}

/// @brief Compute C = alpha * A @ B + beta * C for row-major Float16 inputs and Float32 output.
[[nodiscard]] inline status gemm_f16_f32(
    const buffer& a,
    const buffer& b,
    buffer& c,
    size_t m,
    size_t n,
    size_t k,
    float alpha,
    float beta,
    const queue& q) noexcept
{
    if (!q.is_valid()) {
        return status(error_code::invalid_handle, "linalg::gemm_f16_f32 queue");
    }
    if (detail::mul_overflows(m, k) ||
        detail::mul_overflows(k, n) ||
        detail::mul_overflows(m, n)) {
        return status(error_code::overflow_error, "linalg::gemm_f16_f32 shape");
    }
    auto sa = detail::require_valid_float16(a, m * k, "linalg::gemm_f16_f32 A");
    if (!sa) {
        return sa;
    }
    auto sb = detail::require_valid_float16(b, k * n, "linalg::gemm_f16_f32 B");
    if (!sb) {
        return sb;
    }
    auto sc = detail::require_valid_float32(c, m * n, "linalg::gemm_f16_f32 C");
    if (!sc) {
        return sc;
    }
    return detail::success_or_error(
        navatala_gpu_gemm_f16_f32(
            a.handle(), b.handle(), c.handle(), m, n, k, alpha, beta, q.handle()),
        "linalg::gemm_f16_f32");
}

/// @brief Compute C = alpha * op(A) @ op(B) + beta * C for row-major Float16 inputs and Float32 output.
[[nodiscard]] inline status gemm_f16_f32_ex(
    const buffer& a,
    const buffer& b,
    buffer& c,
    size_t m,
    size_t n,
    size_t k,
    float alpha,
    float beta,
    matrix_transpose trans_a,
    matrix_transpose trans_b,
    const queue& q) noexcept
{
    if (!q.is_valid()) {
        return status(error_code::invalid_handle, "linalg::gemm_f16_f32_ex queue");
    }
    if (detail::mul_overflows(m, k) ||
        detail::mul_overflows(k, n) ||
        detail::mul_overflows(m, n)) {
        return status(error_code::overflow_error, "linalg::gemm_f16_f32_ex shape");
    }
    auto sa = detail::require_valid_float16(a, m * k, "linalg::gemm_f16_f32_ex A");
    if (!sa) {
        return sa;
    }
    auto sb = detail::require_valid_float16(b, k * n, "linalg::gemm_f16_f32_ex B");
    if (!sb) {
        return sb;
    }
    auto sc = detail::require_valid_float32(c, m * n, "linalg::gemm_f16_f32_ex C");
    if (!sc) {
        return sc;
    }
    return detail::success_or_error(
        navatala_gpu_gemm_f16_f32_ex(
            a.handle(), b.handle(), c.handle(), m, n, k, alpha, beta,
            detail::to_ffi(trans_a), detail::to_ffi(trans_b), q.handle()),
        "linalg::gemm_f16_f32_ex");
}

/// @brief Strided-batched Float16-input/Float32-output GEMM.
[[nodiscard]] inline status gemm_f16_f32_strided_batched(
    const buffer& a,
    const buffer& b,
    buffer& c,
    size_t m,
    size_t n,
    size_t k,
    float alpha,
    float beta,
    matrix_transpose trans_a,
    matrix_transpose trans_b,
    size_t stride_a,
    size_t stride_b,
    size_t stride_c,
    size_t batch_count,
    const queue& q) noexcept
{
    if (!q.is_valid()) {
        return status(error_code::invalid_handle, "linalg::gemm_f16_f32_strided_batched queue");
    }
    if (detail::mul_overflows(m, k) ||
        detail::mul_overflows(k, n) ||
        detail::mul_overflows(m, n)) {
        return status(error_code::overflow_error, "linalg::gemm_f16_f32_strided_batched shape");
    }
    const size_t a_matrix_elements = m * k;
    const size_t b_matrix_elements = k * n;
    const size_t c_matrix_elements = m * n;
    size_t a_extent = 0;
    size_t b_extent = 0;
    size_t c_extent = 0;
    if (detail::strided_extent_overflows(a_matrix_elements, stride_a, batch_count, true, &a_extent) ||
        detail::strided_extent_overflows(b_matrix_elements, stride_b, batch_count, true, &b_extent) ||
        detail::strided_extent_overflows(c_matrix_elements, stride_c, batch_count, false, &c_extent)) {
        return status(error_code::overflow_error, "linalg::gemm_f16_f32_strided_batched stride");
    }
    auto sa = detail::require_valid_float16(a, a_extent, "linalg::gemm_f16_f32_strided_batched A");
    if (!sa) {
        return sa;
    }
    auto sb = detail::require_valid_float16(b, b_extent, "linalg::gemm_f16_f32_strided_batched B");
    if (!sb) {
        return sb;
    }
    auto sc = detail::require_valid_float32(c, c_extent, "linalg::gemm_f16_f32_strided_batched C");
    if (!sc) {
        return sc;
    }
    return detail::success_or_error(
        navatala_gpu_gemm_f16_f32_strided_batched(
            a.handle(), b.handle(), c.handle(), m, n, k, alpha, beta,
            detail::to_ffi(trans_a), detail::to_ffi(trans_b),
            stride_a, stride_b, stride_c, batch_count, q.handle()),
        "linalg::gemm_f16_f32_strided_batched");
}

/// @brief Compute the Euclidean norm of a Float32 vector into a scalar buffer.
[[nodiscard]] inline status nrm2(
    const buffer& x,
    buffer& result,
    size_t element_count,
    const queue& q) noexcept
{
    if (!q.is_valid()) {
        return status(error_code::invalid_handle, "linalg::nrm2 queue");
    }
    auto sx = detail::require_valid_float32(x, element_count, "linalg::nrm2 x");
    if (!sx) {
        return sx;
    }
    auto sr = detail::require_valid_float32(result, 1, "linalg::nrm2 result");
    if (!sr) {
        return sr;
    }
    return detail::success_or_error(
        navatala_gpu_nrm2_f32(x.handle(), result.handle(), element_count, q.handle()),
        "linalg::nrm2");
}

} // namespace navatala::linalg

namespace navatala::sparse {

/// @brief Compute y = A * x for a Float32 CSR matrix with UInt32 indices.
[[nodiscard]] inline status csr_spmv(
    const buffer& row_offsets,
    const buffer& col_indices,
    const buffer& values,
    const buffer& x,
    buffer& y,
    size_t rows,
    size_t cols,
    size_t nnz,
    const queue& q) noexcept
{
    if (!q.is_valid()) {
        return status(error_code::invalid_handle, "sparse::csr_spmv queue");
    }
    if (rows == std::numeric_limits<size_t>::max()) {
        return status(error_code::overflow_error, "sparse::csr_spmv rows");
    }
    auto srow = linalg::detail::require_valid_uint32(
        row_offsets, rows + 1, "sparse::csr_spmv row_offsets");
    if (!srow) {
        return srow;
    }
    auto scol = linalg::detail::require_valid_uint32(
        col_indices, nnz, "sparse::csr_spmv col_indices");
    if (!scol) {
        return scol;
    }
    auto sval = linalg::detail::require_valid_float32(
        values, nnz, "sparse::csr_spmv values");
    if (!sval) {
        return sval;
    }
    auto sx = linalg::detail::require_valid_float32(x, cols, "sparse::csr_spmv x");
    if (!sx) {
        return sx;
    }
    auto sy = linalg::detail::require_valid_float32(y, rows, "sparse::csr_spmv y");
    if (!sy) {
        return sy;
    }
    return linalg::detail::success_or_error(
        navatala_gpu_csr_spmv_f32(
            row_offsets.handle(), col_indices.handle(), values.handle(),
            x.handle(), y.handle(), rows, cols, nnz, q.handle()),
        "sparse::csr_spmv");
}

} // namespace navatala::sparse
