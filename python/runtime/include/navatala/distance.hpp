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

// Navatala C++ Wrapper Layer - Distance Computations
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"
#include "error.hpp"
#include "types.hpp"
#include "buffer.hpp"
#include "dataset.hpp"
#include "resource.hpp"
#include "index.hpp"
#include "navatala_ffi.h"

/// @file navatala/distance.hpp
/// @brief GPU-accelerated distance computations.
///
/// This header provides:
/// - Pairwise distance computation
/// - Fused L2 nearest neighbor search
/// - Distance matrix operations

namespace navatala {

/// @brief Namespace for distance computation utilities.
namespace distance {

// ============================================================================
// Pairwise Distance
// ============================================================================

/// @brief Computes pairwise distances between two sets of vectors.
///
/// For X [m, d] and Y [n, d], computes D [m, n] where D[i,j] = dist(X[i], Y[j]).
///
/// @param x First set of vectors [m, d].
/// @param y Second set of vectors [n, d].
/// @param metric Distance metric to use.
/// @param res GPU resources.
/// @return An expected<buffer> containing the distance matrix [m, n] or an error.
///
/// Example:
/// @code
///   auto D = distance::pairwise(X, Y, metric_type::l2, res);
///   // D is [X.n_vectors(), Y.n_vectors()] matrix
/// @endcode
[[nodiscard]] inline result<buffer> pairwise(
    const dataset& x,
    const dataset& y,
    metric_type metric,
    resources& res)
{
    if (!x.is_valid() || !y.is_valid()) {
        return result<buffer>(error_code::invalid_handle, "input datasets invalid");
    }

    if (x.dims() != y.dims()) {
        return result<buffer>(error_code::invalid_param, "dimension mismatch");
    }

    size_t m = x.n_vectors();
    size_t n = y.n_vectors();

    // Create output buffer
    auto out = buffer::create(res, {m, n}, data_type::float32);
    if (!out.has_value()) {
        return result<buffer>(out.error(), "failed to create output buffer");
    }

    // Call FFI
    NavatalaErrorCode err = navatala_pairwise_distance(
        x.device_ptr(), m,
        y.device_ptr(), n,
        x.dims(),
        static_cast<int>(metric),
        out.value().data(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(static_cast<error_code>(err), "pairwise distance failed");
    }

    return out;
}

/// @brief Computes pairwise L2 (Euclidean) distances.
[[nodiscard]] inline result<buffer> pairwise_l2(
    const dataset& x,
    const dataset& y,
    resources& res)
{
    return pairwise(x, y, metric_type::l2, res);
}

/// @brief Computes pairwise inner product distances.
[[nodiscard]] inline result<buffer> pairwise_inner_product(
    const dataset& x,
    const dataset& y,
    resources& res)
{
    return pairwise(x, y, metric_type::inner_product, res);
}

/// @brief Computes pairwise cosine distances.
[[nodiscard]] inline result<buffer> pairwise_cosine(
    const dataset& x,
    const dataset& y,
    resources& res)
{
    return pairwise(x, y, metric_type::cosine, res);
}

// ============================================================================
// Fused L2 Nearest Neighbor
// ============================================================================

/// @brief Fused L2 distance computation with argmin reduction.
///
/// Efficiently computes both the minimum L2 distance and the index
/// of the nearest neighbor in a single kernel.
///
/// @param x Query vectors [m, d].
/// @param y Database vectors [n, d].
/// @param res GPU resources.
/// @return Pair of (min_distances [m], argmin_indices [m]) or error.
[[nodiscard]] inline result<std::pair<buffer, buffer>> fused_l2_nn(
    const dataset& x,
    const dataset& y,
    resources& res)
{
    if (!x.is_valid() || !y.is_valid()) {
        return result<std::pair<buffer, buffer>>(
            error_code::invalid_handle, "input datasets invalid");
    }

    if (x.dims() != y.dims()) {
        return result<std::pair<buffer, buffer>>(
            error_code::invalid_param, "dimension mismatch");
    }

    size_t m = x.n_vectors();
    size_t n = y.n_vectors();

    // Create output buffers
    auto min_dists = buffer::create(res, {m}, data_type::float32);
    if (!min_dists.has_value()) {
        return result<std::pair<buffer, buffer>>(
            min_dists.error(), "failed to create distance buffer");
    }

    auto argmin = buffer::create(res, {m}, data_type::int32);
    if (!argmin.has_value()) {
        return result<std::pair<buffer, buffer>>(
            argmin.error(), "failed to create index buffer");
    }

    // Call FFI
    NavatalaErrorCode err = navatala_fused_l2_nn(
        x.device_ptr(), m,
        y.device_ptr(), n,
        x.dims(),
        min_dists.value().data(),
        argmin.value().data(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<std::pair<buffer, buffer>>(
            static_cast<error_code>(err), "fused L2 NN failed");
    }

    return result<std::pair<buffer, buffer>>(
        std::make_pair(std::move(min_dists.value()), std::move(argmin.value())));
}

// ============================================================================
// Distance Matrix Operations
// ============================================================================

/// @brief Computes row-wise minimum of a distance matrix.
/// @param distances Distance matrix [m, n].
/// @param res GPU resources.
/// @return Buffer with row-wise minima [m] or error.
[[nodiscard]] inline result<buffer> row_min(
    const buffer& distances,
    resources& res)
{
    if (!distances.is_valid()) {
        return result<buffer>(error_code::invalid_handle, "distances buffer invalid");
    }

    if (distances.shape().size() != 2) {
        return result<buffer>(error_code::invalid_param, "expected 2D distance matrix");
    }

    size_t m = distances.shape()[0];

    auto mins = buffer::create(res, {m}, data_type::float32);
    if (!mins.has_value()) {
        return mins;
    }

    NavatalaErrorCode err = navatala_distance_row_min(
        distances.data(),
        m,
        distances.shape()[1],
        mins.value().data(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(static_cast<error_code>(err), "row_min failed");
    }

    return mins;
}

/// @brief Computes row-wise argmin of a distance matrix.
/// @param distances Distance matrix [m, n].
/// @param res GPU resources.
/// @return Buffer with row-wise argmin indices [m] or error.
[[nodiscard]] inline result<buffer> row_argmin(
    const buffer& distances,
    resources& res)
{
    if (!distances.is_valid()) {
        return result<buffer>(error_code::invalid_handle, "distances buffer invalid");
    }

    if (distances.shape().size() != 2) {
        return result<buffer>(error_code::invalid_param, "expected 2D distance matrix");
    }

    size_t m = distances.shape()[0];

    auto argmins = buffer::create(res, {m}, data_type::int32);
    if (!argmins.has_value()) {
        return argmins;
    }

    NavatalaErrorCode err = navatala_distance_row_argmin(
        distances.data(),
        m,
        distances.shape()[1],
        argmins.value().data(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(static_cast<error_code>(err), "row_argmin failed");
    }

    return argmins;
}

/// @brief Computes row-wise top-k of a distance matrix.
/// @param distances Distance matrix [m, n].
/// @param k Number of top elements per row.
/// @param largest If true, returns k largest; if false, returns k smallest.
/// @param res GPU resources.
/// @return Pair of (values [m, k], indices [m, k]) or error.
[[nodiscard]] inline result<std::pair<buffer, buffer>> row_topk(
    const buffer& distances,
    size_t k,
    bool largest,
    resources& res)
{
    if (!distances.is_valid()) {
        return result<std::pair<buffer, buffer>>(
            error_code::invalid_handle, "distances buffer invalid");
    }

    if (distances.shape().size() != 2) {
        return result<std::pair<buffer, buffer>>(
            error_code::invalid_param, "expected 2D distance matrix");
    }

    size_t m = distances.shape()[0];
    size_t n = distances.shape()[1];

    if (k > n) {
        return result<std::pair<buffer, buffer>>(
            error_code::invalid_param, "k exceeds number of columns");
    }

    auto values = buffer::create(res, {m, k}, data_type::float32);
    if (!values.has_value()) {
        return result<std::pair<buffer, buffer>>(
            values.error(), "failed to create values buffer");
    }

    auto indices = buffer::create(res, {m, k}, data_type::int32);
    if (!indices.has_value()) {
        return result<std::pair<buffer, buffer>>(
            indices.error(), "failed to create indices buffer");
    }

    NavatalaErrorCode err = navatala_distance_row_topk(
        distances.data(),
        m, n, k,
        largest ? 1 : 0,
        values.value().data(),
        indices.value().data(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<std::pair<buffer, buffer>>(
            static_cast<error_code>(err), "row_topk failed");
    }

    return result<std::pair<buffer, buffer>>(
        std::make_pair(std::move(values.value()), std::move(indices.value())));
}

// ============================================================================
// Vector Normalization
// ============================================================================

/// @brief Normalizes vectors to unit length (L2 norm).
/// @param data Dataset to normalize in-place.
/// @param res GPU resources.
/// @return Status indicating success or failure.
[[nodiscard]] inline status normalize_l2(
    dataset& data,
    resources& res)
{
    if (!data.is_valid()) {
        return status(error_code::invalid_handle, "dataset invalid");
    }

    NavatalaErrorCode err = navatala_normalize_l2(
        data.device_ptr(),
        data.n_vectors(),
        data.dims(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return status(static_cast<error_code>(err), "L2 normalization failed");
    }

    return status();
}

/// @brief Computes L2 norms of vectors.
/// @param data Input dataset.
/// @param res GPU resources.
/// @return Buffer with norms [n_vectors] or error.
[[nodiscard]] inline result<buffer> compute_norms(
    const dataset& data,
    resources& res)
{
    if (!data.is_valid()) {
        return result<buffer>(error_code::invalid_handle, "dataset invalid");
    }

    auto norms = buffer::create(res, {data.n_vectors()}, data_type::float32);
    if (!norms.has_value()) {
        return norms;
    }

    NavatalaErrorCode err = navatala_compute_norms(
        data.device_ptr(),
        data.n_vectors(),
        data.dims(),
        norms.value().data(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(static_cast<error_code>(err), "compute norms failed");
    }

    return norms;
}

} // namespace distance
} // namespace navatala
