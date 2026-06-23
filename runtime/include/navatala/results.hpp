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

// Navatala C++ Wrapper Layer - Search Results
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"
#include "error.hpp"
#include "types.hpp"
#include "buffer.hpp"
#include "resource.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

/// @file navatala/results.hpp
/// @brief Container for k-NN search results.
///
/// This header provides:
/// - search_results: GPU-resident storage for neighbor indices and distances
/// - Host copy utilities for result retrieval
/// - Batch result access for multi-query searches

namespace navatala {

// ============================================================================
// Search Results Class
// ============================================================================

/// @brief Container for k-NN search results.
///
/// Holds the indices of nearest neighbors and their distances for a batch
/// of queries. Results are stored in row-major order where each row contains
/// the k neighbors for one query vector.
///
/// Layout: For n_queries queries with k neighbors each:
/// - indices: [n_queries, k] matrix of neighbor indices (int64 or uint32)
/// - distances: [n_queries, k] matrix of distances (float32)
///
/// Example:
/// @code
///   // After performing a search
///   auto results = index.search(queries, k, res);
///
///   // Get results for query 0
///   auto neighbors = results.value().indices_for_query(0, res);
///   auto dists = results.value().distances_for_query(0, res);
///
///   // Or get all results at once
///   auto all_indices = results.value().indices_to_host(res);
///   auto all_distances = results.value().distances_to_host(res);
/// @endcode
///
/// @thread_safety Results can be read concurrently from multiple threads
///                after the search operation completes.
class search_results {
public:
    // -------------------- Factory Methods --------------------

    /// Creates a search_results container with pre-allocated storage.
    /// @param res The GPU resources.
    /// @param n_queries Number of query vectors.
    /// @param k Number of neighbors per query.
    /// @param index_type Data type for indices (uint32 or int64).
    /// @return An expected<search_results> or an error.
    [[nodiscard]] static result<search_results> create(
        resources& res,
        size_t n_queries,
        size_t k,
        data_type index_type = data_type::uint32);

    /// Wraps existing device buffers as search results.
    /// @param indices Buffer containing neighbor indices [n_queries, k].
    /// @param distances Buffer containing distances [n_queries, k].
    /// @param n_queries Number of queries.
    /// @param k Number of neighbors.
    /// @return An expected<search_results> or an error.
    [[nodiscard]] static result<search_results> wrap(
        buffer&& indices,
        buffer&& distances,
        size_t n_queries,
        size_t k);

    // -------------------- Constructors/Destructor --------------------

    /// Default constructor creates an invalid search_results.
    search_results() noexcept;

    /// Destructor.
    ~search_results() noexcept = default;

    // Non-copyable
    search_results(const search_results&) = delete;
    search_results& operator=(const search_results&) = delete;

    // Movable
    search_results(search_results&& other) noexcept;
    search_results& operator=(search_results&& other) noexcept;

    // -------------------- Accessors --------------------

    /// Returns the number of queries.
    [[nodiscard]] size_t n_queries() const noexcept { return n_queries_; }

    /// Returns the number of neighbors per query (k).
    [[nodiscard]] size_t k() const noexcept { return k_; }

    /// Returns the total number of result pairs (n_queries * k).
    [[nodiscard]] size_t size() const noexcept { return n_queries_ * k_; }

    /// Returns true if the results are valid.
    [[nodiscard]] bool is_valid() const noexcept {
        return indices_.is_valid() && distances_.is_valid();
    }

    /// Explicit bool conversion.
    explicit operator bool() const noexcept { return is_valid(); }

    /// Returns the data type of indices.
    [[nodiscard]] data_type index_dtype() const noexcept { return indices_.dtype(); }

    /// Returns the underlying indices buffer.
    [[nodiscard]] const buffer& indices_buffer() const noexcept { return indices_; }

    /// Returns the underlying distances buffer.
    [[nodiscard]] const buffer& distances_buffer() const noexcept { return distances_; }

    /// Returns mutable access to indices buffer.
    [[nodiscard]] buffer& indices_buffer() noexcept { return indices_; }

    /// Returns mutable access to distances buffer.
    [[nodiscard]] buffer& distances_buffer() noexcept { return distances_; }

    // -------------------- Host Copy Operations --------------------

    /// Copies all indices to host memory.
    /// @tparam T Index type (uint32_t or int64_t).
    /// @param res Resources for synchronization.
    /// @return Vector of all indices in row-major order.
    template<typename T = uint32_t>
    [[nodiscard]] std::vector<T> indices_to_host(resources& res) const {
        return indices_.to_host<T>(res);
    }

    /// Copies all distances to host memory.
    /// @param res Resources for synchronization.
    /// @return Vector of all distances in row-major order.
    [[nodiscard]] std::vector<float> distances_to_host(resources& res) const {
        return distances_.to_host<float>(res);
    }

    /// Copies indices for a single query to host.
    /// @tparam T Index type.
    /// @param query_idx The query index (0-based).
    /// @param res Resources for synchronization.
    /// @return Vector of k neighbor indices for the specified query.
    template<typename T = uint32_t>
    [[nodiscard]] std::vector<T> indices_for_query(size_t query_idx, resources& res) const {
        if (query_idx >= n_queries_) {
            return {};
        }
        // For now, copy full buffer and slice (correct but wasteful for large result sets)
        auto all = indices_to_host<T>(res);
        std::vector<T> result(k_);
        std::copy(all.begin() + query_idx * k_,
                  all.begin() + (query_idx + 1) * k_,
                  result.begin());
        return result;
    }

    /// Copies distances for a single query to host.
    /// @param query_idx The query index (0-based).
    /// @param res Resources for synchronization.
    /// @return Vector of k distances for the specified query.
    [[nodiscard]] std::vector<float> distances_for_query(size_t query_idx, resources& res) const {
        if (query_idx >= n_queries_) {
            return {};
        }
        std::vector<float> result(k_);
        // Note: This is a simplified implementation; proper slicing would be better
        auto all = distances_to_host(res);
        std::copy(all.begin() + query_idx * k_,
                  all.begin() + (query_idx + 1) * k_,
                  result.begin());
        return result;
    }

    // -------------------- Result Analysis --------------------

    /// Returns the index of the nearest neighbor for a query.
    /// @tparam T Index type.
    /// @param query_idx The query index.
    /// @param res Resources for synchronization.
    /// @return The index of the nearest neighbor, or -1 on error.
    template<typename T = uint32_t>
    [[nodiscard]] T nearest_neighbor(size_t query_idx, resources& res) const {
        auto indices = indices_for_query<T>(query_idx, res);
        return indices.empty() ? static_cast<T>(-1) : indices[0];
    }

    /// Returns the distance to the nearest neighbor for a query.
    /// @param query_idx The query index.
    /// @param res Resources for synchronization.
    /// @return The distance to the nearest neighbor, or infinity on error.
    [[nodiscard]] float nearest_distance(size_t query_idx, resources& res) const {
        auto dists = distances_for_query(query_idx, res);
        return dists.empty() ? std::numeric_limits<float>::infinity() : dists[0];
    }

private:
    // Private constructor
    search_results(buffer indices, buffer distances, size_t n_queries, size_t k);

    buffer indices_;
    buffer distances_;
    size_t n_queries_;
    size_t k_;
};

// ============================================================================
// Inline Implementations
// ============================================================================

inline search_results::search_results() noexcept
    : indices_()
    , distances_()
    , n_queries_(0)
    , k_(0)
{}

inline search_results::search_results(buffer indices, buffer distances,
                                       size_t n_queries, size_t k)
    : indices_(std::move(indices))
    , distances_(std::move(distances))
    , n_queries_(n_queries)
    , k_(k)
{}

inline search_results::search_results(search_results&& other) noexcept
    : indices_(std::move(other.indices_))
    , distances_(std::move(other.distances_))
    , n_queries_(other.n_queries_)
    , k_(other.k_)
{
    other.n_queries_ = 0;
    other.k_ = 0;
}

inline search_results& search_results::operator=(search_results&& other) noexcept {
    if (this != &other) {
        indices_ = std::move(other.indices_);
        distances_ = std::move(other.distances_);
        n_queries_ = other.n_queries_;
        k_ = other.k_;

        other.n_queries_ = 0;
        other.k_ = 0;
    }
    return *this;
}

inline result<search_results> search_results::create(
    resources& res,
    size_t n_queries,
    size_t k,
    data_type index_type)
{
    if (n_queries == 0 || k == 0) {
        return result<search_results>(error_code::invalid_param,
                                       "n_queries and k must be positive");
    }

    // Create indices buffer
    auto indices = buffer::create(res, {n_queries, k}, index_type);
    if (!indices.has_value()) {
        return result<search_results>(indices.error(), "failed to create indices buffer");
    }

    // Create distances buffer
    auto distances = buffer::create(res, {n_queries, k}, data_type::float32);
    if (!distances.has_value()) {
        return result<search_results>(distances.error(), "failed to create distances buffer");
    }

    return result<search_results>(
        search_results(std::move(indices.value()), std::move(distances.value()),
                       n_queries, k));
}

inline result<search_results> search_results::wrap(
    buffer&& indices,
    buffer&& distances,
    size_t n_queries,
    size_t k)
{
    if (!indices.is_valid() || !distances.is_valid()) {
        return result<search_results>(error_code::invalid_handle,
                                       "indices or distances buffer is invalid");
    }

    if (indices.num_elements() < n_queries * k) {
        return result<search_results>(error_code::invalid_param,
                                       "indices buffer too small");
    }

    if (distances.num_elements() < n_queries * k) {
        return result<search_results>(error_code::invalid_param,
                                       "distances buffer too small");
    }

    return result<search_results>(
        search_results(std::move(indices), std::move(distances), n_queries, k));
}

} // namespace navatala
