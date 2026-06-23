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

// Navatala C++ Wrapper Layer - Vector Search Indices
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"
#include "error.hpp"
#include "types.hpp"
#include "buffer.hpp"
#include "dataset.hpp"
#include "resource.hpp"
#include "results.hpp"

#include <memory>
#include <optional>
#include <string>

/// @file navatala/index.hpp
/// @brief GPU-accelerated vector search index types.
///
/// This header provides:
/// - cagra_index: Graph-based approximate nearest neighbor search
/// - ivfpq_index: Inverted file with product quantization
/// - ivfflat_index: Inverted file with flat (exact) search
/// - Common build and search parameter structures

// Forward declare FFI types
struct NavatalaVectorIndex;

namespace navatala {

// ============================================================================
// Distance Metric Type
// ============================================================================

/// @brief Distance metrics for vector search.
enum class metric_type : uint32_t {
    l2 = 0,           ///< Euclidean (L2) distance
    inner_product = 1, ///< Inner product (higher = more similar)
    cosine = 2         ///< Cosine similarity
};

/// @brief Returns the name of a metric type.
[[nodiscard]] constexpr const char* metric_name(metric_type m) noexcept {
    switch (m) {
        case metric_type::l2:            return "l2";
        case metric_type::inner_product: return "inner_product";
        case metric_type::cosine:        return "cosine";
        default:                         return "unknown";
    }
}

// ============================================================================
// Index Type Enumeration
// ============================================================================

/// @brief Types of vector search indices.
enum class knn_index_type : uint32_t {
    cagra = 0,    ///< CAGRA graph-based index
    ivfpq = 1,    ///< IVF with product quantization
    ivfflat = 2,  ///< IVF with flat (exact) search
    brute_force = 3 ///< Exact brute-force search
};

/// @brief Returns the name of an index type.
[[nodiscard]] constexpr const char* knn_index_type_name(knn_index_type t) noexcept {
    switch (t) {
        case knn_index_type::cagra:       return "cagra";
        case knn_index_type::ivfpq:       return "ivfpq";
        case knn_index_type::ivfflat:     return "ivfflat";
        case knn_index_type::brute_force: return "brute_force";
        default:                      return "unknown";
    }
}

// ============================================================================
// Build Parameters
// ============================================================================

/// @brief Parameters for building a CAGRA index.
struct cagra_build_params {
    /// Number of neighbors in the graph (degree).
    size_t graph_degree = 64;

    /// Intermediate graph degree during construction.
    size_t intermediate_graph_degree = 128;

    /// Build algorithm: 0 = IVF-PQ based, 1 = NN-Descent based.
    int build_algo = 0;

    /// Number of iterations for NN-Descent.
    size_t nn_descent_niter = 20;

    /// Distance metric.
    metric_type metric = metric_type::l2;
};

/// @brief Parameters for building an IVF-PQ index.
struct ivfpq_build_params {
    /// Number of inverted lists (clusters).
    size_t n_lists = 1024;

    /// Number of subquantizers (PQ dimension).
    size_t pq_dim = 0;  // 0 = auto (dims / pq_bits)

    /// Bits per PQ code (4 or 8).
    size_t pq_bits = 8;

    /// K-means iterations for training.
    size_t kmeans_n_iters = 20;

    /// Fraction of data to use for training (0.0-1.0).
    float kmeans_trainset_fraction = 0.5f;

    /// Distance metric.
    metric_type metric = metric_type::l2;
};

/// @brief Parameters for building an IVF-Flat index.
struct ivfflat_build_params {
    /// Number of inverted lists (clusters).
    size_t n_lists = 1024;

    /// K-means iterations for training.
    size_t kmeans_n_iters = 20;

    /// Fraction of data to use for training (0.0-1.0).
    float kmeans_trainset_fraction = 0.5f;

    /// Distance metric.
    metric_type metric = metric_type::l2;
};

// ============================================================================
// Search Parameters
// ============================================================================

/// @brief Parameters for CAGRA search.
struct cagra_search_params {
    /// Maximum number of graph traversal iterations.
    size_t max_iterations = 0;  // 0 = auto

    /// Internal topk for search (affects accuracy vs speed).
    size_t itopk_size = 64;

    /// Search width (number of entry points to explore).
    size_t search_width = 1;

    /// Minimum iterations before early stopping.
    size_t min_iterations = 0;

    /// Number of threads per block (0 = auto).
    size_t thread_block_size = 0;

    /// Hashmap mode: 0 = hash, 1 = small, 2 = auto.
    int hashmap_mode = 2;

    /// Hashmap load factor.
    float hashmap_max_fill_rate = 0.5f;

    /// Number of random restarts.
    size_t num_random_samplings = 1;

    /// Random seed for sampling.
    uint64_t rand_xor_mask = 0x128394;
};

/// @brief Parameters for IVF-PQ search.
struct ivfpq_search_params {
    /// Number of probes (clusters to search).
    size_t n_probes = 20;

    /// LUT data type: 0 = fp32, 1 = fp16, 2 = fp8.
    int lut_dtype = 0;

    /// Internal distance data type.
    int internal_distance_dtype = 0;

    /// Preferred shmem carveout (0.0 - 1.0).
    float preferred_shmem_carveout = 1.0f;
};

/// @brief Parameters for IVF-Flat search.
struct ivfflat_search_params {
    /// Number of probes (clusters to search).
    size_t n_probes = 20;
};

// ============================================================================
// Base Index Interface
// ============================================================================

/// @brief Base class for vector search indices.
///
/// Provides common interface for all index types.
class index_base {
public:
    virtual ~index_base() = default;

    /// Returns the index type.
    [[nodiscard]] virtual knn_index_type type() const noexcept = 0;

    /// Returns the dimensionality of vectors.
    [[nodiscard]] virtual size_t dims() const noexcept = 0;

    /// Returns the number of vectors in the index.
    [[nodiscard]] virtual size_t size() const noexcept = 0;

    /// Returns true if the index is trained/built.
    [[nodiscard]] virtual bool is_trained() const noexcept = 0;

    /// Returns the distance metric.
    [[nodiscard]] virtual metric_type metric() const noexcept = 0;

    /// Returns true if the index is valid.
    [[nodiscard]] virtual bool is_valid() const noexcept = 0;
};

// ============================================================================
// CAGRA Index
// ============================================================================

/// @brief CAGRA graph-based approximate nearest neighbor index.
///
/// CAGRA (CUDA Accelerated Graph-based Approximate nearest neighbor search)
/// builds a proximity graph and uses graph traversal for fast k-NN search.
///
/// Example:
/// @code
///   auto res = resources::create().value();
///
///   // Build index from dataset
///   auto ds = dataset::create(res, vectors, n_vectors, dims, data_type::float32).value();
///   cagra_build_params build_params;
///   build_params.graph_degree = 64;
///
///   auto index = cagra_index::build(res, ds, build_params).value();
///
///   // Search
///   auto queries = dataset::create(res, query_vecs, n_queries, dims, data_type::float32).value();
///   cagra_search_params search_params;
///   auto results = index.search(queries, 10, search_params, res).value();
/// @endcode
class cagra_index : public index_base {
public:
    // -------------------- Factory Methods --------------------

    /// Builds a CAGRA index from a dataset.
    /// @param res GPU resources.
    /// @param data Dataset of vectors to index.
    /// @param params Build parameters.
    /// @return An expected<cagra_index> or an error.
    [[nodiscard]] static result<cagra_index> build(
        resources& res,
        const dataset& data,
        const cagra_build_params& params = {});

    /// Loads a CAGRA index from file.
    /// @param res GPU resources.
    /// @param path Path to the index file.
    /// @return An expected<cagra_index> or an error.
    [[nodiscard]] static result<cagra_index> load(
        resources& res,
        const std::string& path);

    // -------------------- Constructors/Destructor --------------------

    /// Default constructor creates an invalid index.
    cagra_index() noexcept;

    /// Destructor.
    ~cagra_index() noexcept override;

    // Non-copyable
    cagra_index(const cagra_index&) = delete;
    cagra_index& operator=(const cagra_index&) = delete;

    // Movable
    cagra_index(cagra_index&& other) noexcept;
    cagra_index& operator=(cagra_index&& other) noexcept;

    // -------------------- Index Base Interface --------------------

    [[nodiscard]] knn_index_type type() const noexcept override { return knn_index_type::cagra; }
    [[nodiscard]] size_t dims() const noexcept override { return dims_; }
    [[nodiscard]] size_t size() const noexcept override { return n_vectors_; }
    [[nodiscard]] bool is_trained() const noexcept override { return handle_ != nullptr; }
    [[nodiscard]] metric_type metric() const noexcept override { return metric_; }
    [[nodiscard]] bool is_valid() const noexcept override { return handle_ != nullptr; }

    // -------------------- CAGRA-Specific Methods --------------------

    /// Returns the graph degree.
    [[nodiscard]] size_t graph_degree() const noexcept { return graph_degree_; }

    /// Performs k-NN search.
    /// @param queries Query vectors.
    /// @param k Number of neighbors to find.
    /// @param params Search parameters.
    /// @param res GPU resources.
    /// @return An expected<search_results> or an error.
    [[nodiscard]] result<search_results> search(
        const dataset& queries,
        size_t k,
        const cagra_search_params& params,
        resources& res) const;

    /// Performs k-NN search with default parameters.
    [[nodiscard]] result<search_results> search(
        const dataset& queries,
        size_t k,
        resources& res) const {
        return search(queries, k, cagra_search_params{}, res);
    }

    /// Saves the index to file.
    /// @param path Output file path.
    /// @return Status indicating success or failure.
    [[nodiscard]] status save(const std::string& path) const;

    /// Returns the raw FFI handle.
    [[nodiscard]] NavatalaVectorIndex* handle() const noexcept { return handle_; }

private:
    // Private constructor
    cagra_index(NavatalaVectorIndex* handle, resources* res,
                size_t dims, size_t n_vectors, size_t graph_degree,
                metric_type metric);

    NavatalaVectorIndex* handle_;
    resources* parent_resources_;
    size_t dims_;
    size_t n_vectors_;
    size_t graph_degree_;
    metric_type metric_;
};

// ============================================================================
// IVF-PQ Index
// ============================================================================

/// @brief IVF-PQ (Inverted File with Product Quantization) index.
///
/// Combines inverted file indexing with product quantization for
/// memory-efficient approximate nearest neighbor search.
class ivfpq_index : public index_base {
public:
    // -------------------- Factory Methods --------------------

    /// Builds an IVF-PQ index from a dataset.
    [[nodiscard]] static result<ivfpq_index> build(
        resources& res,
        const dataset& data,
        const ivfpq_build_params& params = {});

    /// Loads an IVF-PQ index from file.
    [[nodiscard]] static result<ivfpq_index> load(
        resources& res,
        const std::string& path);

    // -------------------- Constructors/Destructor --------------------

    ivfpq_index() noexcept;
    ~ivfpq_index() noexcept override;

    ivfpq_index(const ivfpq_index&) = delete;
    ivfpq_index& operator=(const ivfpq_index&) = delete;

    ivfpq_index(ivfpq_index&& other) noexcept;
    ivfpq_index& operator=(ivfpq_index&& other) noexcept;

    // -------------------- Index Base Interface --------------------

    [[nodiscard]] knn_index_type type() const noexcept override { return knn_index_type::ivfpq; }
    [[nodiscard]] size_t dims() const noexcept override { return dims_; }
    [[nodiscard]] size_t size() const noexcept override { return n_vectors_; }
    [[nodiscard]] bool is_trained() const noexcept override { return handle_ != nullptr; }
    [[nodiscard]] metric_type metric() const noexcept override { return metric_; }
    [[nodiscard]] bool is_valid() const noexcept override { return handle_ != nullptr; }

    // -------------------- IVF-PQ-Specific Methods --------------------

    /// Returns the number of inverted lists.
    [[nodiscard]] size_t n_lists() const noexcept { return n_lists_; }

    /// Returns the PQ dimension.
    [[nodiscard]] size_t pq_dim() const noexcept { return pq_dim_; }

    /// Returns the PQ bits.
    [[nodiscard]] size_t pq_bits() const noexcept { return pq_bits_; }

    /// Performs k-NN search.
    [[nodiscard]] result<search_results> search(
        const dataset& queries,
        size_t k,
        const ivfpq_search_params& params,
        resources& res) const;

    [[nodiscard]] result<search_results> search(
        const dataset& queries,
        size_t k,
        resources& res) const {
        return search(queries, k, ivfpq_search_params{}, res);
    }

    [[nodiscard]] status save(const std::string& path) const;

    [[nodiscard]] NavatalaVectorIndex* handle() const noexcept { return handle_; }

private:
    ivfpq_index(NavatalaVectorIndex* handle, resources* res,
                size_t dims, size_t n_vectors, size_t n_lists,
                size_t pq_dim, size_t pq_bits, metric_type metric);

    NavatalaVectorIndex* handle_;
    resources* parent_resources_;
    size_t dims_;
    size_t n_vectors_;
    size_t n_lists_;
    size_t pq_dim_;
    size_t pq_bits_;
    metric_type metric_;
};

// ============================================================================
// IVF-Flat Index
// ============================================================================

/// @brief IVF-Flat (Inverted File with Flat search) index.
///
/// Uses inverted file for coarse quantization with exact distance
/// computation within each cluster.
class ivfflat_index : public index_base {
public:
    // -------------------- Factory Methods --------------------

    [[nodiscard]] static result<ivfflat_index> build(
        resources& res,
        const dataset& data,
        const ivfflat_build_params& params = {});

    [[nodiscard]] static result<ivfflat_index> load(
        resources& res,
        const std::string& path);

    // -------------------- Constructors/Destructor --------------------

    ivfflat_index() noexcept;
    ~ivfflat_index() noexcept override;

    ivfflat_index(const ivfflat_index&) = delete;
    ivfflat_index& operator=(const ivfflat_index&) = delete;

    ivfflat_index(ivfflat_index&& other) noexcept;
    ivfflat_index& operator=(ivfflat_index&& other) noexcept;

    // -------------------- Index Base Interface --------------------

    [[nodiscard]] knn_index_type type() const noexcept override { return knn_index_type::ivfflat; }
    [[nodiscard]] size_t dims() const noexcept override { return dims_; }
    [[nodiscard]] size_t size() const noexcept override { return n_vectors_; }
    [[nodiscard]] bool is_trained() const noexcept override { return handle_ != nullptr; }
    [[nodiscard]] metric_type metric() const noexcept override { return metric_; }
    [[nodiscard]] bool is_valid() const noexcept override { return handle_ != nullptr; }

    // -------------------- IVF-Flat-Specific Methods --------------------

    [[nodiscard]] size_t n_lists() const noexcept { return n_lists_; }

    [[nodiscard]] result<search_results> search(
        const dataset& queries,
        size_t k,
        const ivfflat_search_params& params,
        resources& res) const;

    [[nodiscard]] result<search_results> search(
        const dataset& queries,
        size_t k,
        resources& res) const {
        return search(queries, k, ivfflat_search_params{}, res);
    }

    [[nodiscard]] status save(const std::string& path) const;

    [[nodiscard]] NavatalaVectorIndex* handle() const noexcept { return handle_; }

private:
    ivfflat_index(NavatalaVectorIndex* handle, resources* res,
                  size_t dims, size_t n_vectors, size_t n_lists,
                  metric_type metric);

    NavatalaVectorIndex* handle_;
    resources* parent_resources_;
    size_t dims_;
    size_t n_vectors_;
    size_t n_lists_;
    metric_type metric_;
};

// ============================================================================
// Brute Force Index
// ============================================================================

/// @brief Exact brute-force k-NN search.
///
/// Computes exact distances to all vectors. Useful for small datasets
/// or as a baseline for accuracy comparison.
class brute_force_index : public index_base {
public:
    /// Creates a brute-force index from a dataset.
    [[nodiscard]] static result<brute_force_index> build(
        resources& res,
        const dataset& data,
        metric_type metric = metric_type::l2);

    brute_force_index() noexcept;
    ~brute_force_index() noexcept override;

    brute_force_index(const brute_force_index&) = delete;
    brute_force_index& operator=(const brute_force_index&) = delete;

    brute_force_index(brute_force_index&& other) noexcept;
    brute_force_index& operator=(brute_force_index&& other) noexcept;

    [[nodiscard]] knn_index_type type() const noexcept override { return knn_index_type::brute_force; }
    [[nodiscard]] size_t dims() const noexcept override { return dims_; }
    [[nodiscard]] size_t size() const noexcept override { return n_vectors_; }
    [[nodiscard]] bool is_trained() const noexcept override { return data_ != nullptr; }
    [[nodiscard]] metric_type metric() const noexcept override { return metric_; }
    [[nodiscard]] bool is_valid() const noexcept override { return data_ != nullptr; }

    [[nodiscard]] result<search_results> search(
        const dataset& queries,
        size_t k,
        resources& res) const;

private:
    brute_force_index(const dataset* data, resources* res,
                      size_t dims, size_t n_vectors, metric_type metric);

    const dataset* data_;  // Non-owning reference to indexed data
    resources* parent_resources_;
    size_t dims_;
    size_t n_vectors_;
    metric_type metric_;
};

} // namespace navatala
