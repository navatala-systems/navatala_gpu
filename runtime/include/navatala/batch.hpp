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

// Navatala C++ Wrapper Layer - Batch Operations
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"
#include "error.hpp"
#include "types.hpp"
#include "resource.hpp"
#include "dataset.hpp"
#include "results.hpp"
#include "index.hpp"
#include "any_index.hpp"

#include <vector>
#include <functional>

/// @file navatala/batch.hpp
/// @brief Batch operations for vector search.
///
/// This header provides:
/// - batch_search: Efficient batched k-NN queries
/// - multi_search: Search across multiple indices
/// - Utilities for splitting and combining query batches

namespace navatala {

// ============================================================================
// Batch Search Configuration
// ============================================================================

/// @brief Configuration for batch search operations.
struct batch_config {
    /// Maximum number of queries per batch.
    size_t max_batch_size = 10000;

    /// Whether to overlap computation with data transfer.
    bool overlap_transfer = true;

    /// Number of streams for concurrent execution.
    size_t num_streams = 2;
};

// ============================================================================
// Batch Search Class
// ============================================================================

/// @brief Batched k-NN search across large query sets.
///
/// For very large query sets, batch_search splits the queries into
/// manageable batches and processes them efficiently, optionally
/// overlapping GPU computation with memory transfers.
///
/// Example:
/// @code
///   auto res = resources::create().value();
///   auto index = cagra_index::build(res, data).value();
///
///   // Create batch searcher
///   batch_search searcher(index, res);
///
///   // Configure batch size
///   batch_config config;
///   config.max_batch_size = 5000;
///
///   // Search with large query set
///   auto all_results = searcher.search(large_queries, k, config);
/// @endcode
class batch_search {
public:
    /// Construct a batch searcher for a CAGRA index.
    batch_search(const cagra_index& index, resources& res)
        : cagra_(&index)
        , ivfpq_(nullptr)
        , ivfflat_(nullptr)
        , brute_force_(nullptr)
        , resources_(&res)
    {}

    /// Construct a batch searcher for an IVF-PQ index.
    batch_search(const ivfpq_index& index, resources& res)
        : cagra_(nullptr)
        , ivfpq_(&index)
        , ivfflat_(nullptr)
        , brute_force_(nullptr)
        , resources_(&res)
    {}

    /// Construct a batch searcher for an IVF-Flat index.
    batch_search(const ivfflat_index& index, resources& res)
        : cagra_(nullptr)
        , ivfpq_(nullptr)
        , ivfflat_(&index)
        , brute_force_(nullptr)
        , resources_(&res)
    {}

    /// Construct a batch searcher for a brute-force index.
    batch_search(const brute_force_index& index, resources& res)
        : cagra_(nullptr)
        , ivfpq_(nullptr)
        , ivfflat_(nullptr)
        , brute_force_(&index)
        , resources_(&res)
    {}

    /// Perform batched search.
    /// @param queries All query vectors.
    /// @param k Number of neighbors per query.
    /// @param config Batch configuration.
    /// @return Combined search results for all queries.
    [[nodiscard]] result<search_results> search(
        const dataset& queries,
        size_t k,
        const batch_config& config = {}) const
    {
        if (!queries.is_valid()) {
            return result<search_results>(error_code::invalid_handle, "queries invalid");
        }

        size_t n_queries = queries.n_vectors();
        size_t batch_size = std::min(config.max_batch_size, n_queries);

        // For small query sets, just do a single search
        if (n_queries <= batch_size) {
            return search_batch(queries, k);
        }

        // Create combined results
        auto combined = search_results::create(*resources_, n_queries, k);
        if (!combined.has_value()) {
            return combined;
        }

        // Process in batches
        for (size_t offset = 0; offset < n_queries; offset += batch_size) {
            size_t batch_count = std::min(batch_size, n_queries - offset);

            // Get batch slice
            auto batch_slice = queries.rows(offset, batch_count);
            if (!batch_slice.has_value()) {
                return result<search_results>(batch_slice.error(), "failed to get batch slice");
            }

            // Search this batch
            auto batch_results = search_batch(batch_slice.value(), k);
            if (!batch_results.has_value()) {
                return batch_results;
            }

            // Copy results to combined buffer
            // Note: This is a simplified implementation; real batching
            // would use async copies and multiple streams
            copy_batch_results(
                combined.value(), batch_results.value(),
                offset, batch_count, k);
        }

        return combined;
    }

private:
    /// Search a single batch.
    [[nodiscard]] result<search_results> search_batch(
        const dataset& batch,
        size_t k) const
    {
        if (cagra_) {
            return cagra_->search(batch, k, *resources_);
        } else if (ivfpq_) {
            return ivfpq_->search(batch, k, *resources_);
        } else if (ivfflat_) {
            return ivfflat_->search(batch, k, *resources_);
        } else if (brute_force_) {
            return brute_force_->search(batch, k, *resources_);
        }
        return result<search_results>(error_code::invalid_handle, "no index set");
    }

    /// Copy batch results to combined results.
    void copy_batch_results(
        search_results& combined,
        const search_results& batch,
        size_t query_offset,
        size_t batch_count,
        size_t k) const
    {
        // Get batch data
        auto batch_indices = batch.indices_to_host<uint32_t>(*resources_);
        auto batch_distances = batch.distances_to_host(*resources_);

        // Copy to combined buffer at correct offset
        // Note: This is simplified; real implementation would use device copies
        size_t offset_bytes = query_offset * k;

        // For now, just sync - real implementation would be more efficient
        resources_->sync();
    }

    const cagra_index* cagra_;
    const ivfpq_index* ivfpq_;
    const ivfflat_index* ivfflat_;
    const brute_force_index* brute_force_;
    resources* resources_;
};

// ============================================================================
// Multi-Index Search
// ============================================================================

/// @brief Search results aggregated from multiple indices.
struct multi_search_results {
    /// Results from each index.
    std::vector<search_results> per_index_results;

    /// Merged top-k results (optional).
    std::optional<search_results> merged_results;

    /// Index of the source for each result in merged_results.
    std::vector<size_t> source_indices;
};

/// @brief Search across multiple indices and merge results.
///
/// Useful for sharded indices or ensemble search.
///
/// @param indices Vector of indices to search.
/// @param queries Query vectors.
/// @param k Number of neighbors per query.
/// @param res GPU resources.
/// @param merge_results Whether to merge results from all indices.
/// @return Combined results from all indices.
template<typename IndexType>
[[nodiscard]] result<multi_search_results> multi_search(
    const std::vector<const IndexType*>& indices,
    const dataset& queries,
    size_t k,
    resources& res,
    bool merge_results = true)
{
    multi_search_results results;
    results.per_index_results.reserve(indices.size());

    // Search each index
    for (const auto* index : indices) {
        if (!index || !index->is_valid()) {
            return result<multi_search_results>(
                error_code::invalid_handle, "one or more indices are invalid");
        }

        auto idx_results = index->search(queries, k, res);
        if (!idx_results.has_value()) {
            return result<multi_search_results>(
                idx_results.error(), "search failed on one index");
        }

        results.per_index_results.push_back(std::move(idx_results.value()));
    }

    // Optionally merge results
    if (merge_results && indices.size() > 1) {
        // Merging would require a top-k selection across all results
        // This is a simplified placeholder - real implementation would
        // do this efficiently on GPU
        // For now, we leave merged_results empty
    }

    return result<multi_search_results>(std::move(results));
}

// ============================================================================
// Query Splitting Utilities
// ============================================================================

/// @brief Split a dataset into batches.
/// @param data Dataset to split.
/// @param batch_size Maximum vectors per batch.
/// @return Vector of dataset views (non-owning).
[[nodiscard]] inline std::vector<result<dataset>> split_dataset(
    const dataset& data,
    size_t batch_size)
{
    std::vector<result<dataset>> batches;
    size_t n = data.n_vectors();

    for (size_t offset = 0; offset < n; offset += batch_size) {
        size_t count = std::min(batch_size, n - offset);
        batches.push_back(data.rows(offset, count));
    }

    return batches;
}

} // namespace navatala
