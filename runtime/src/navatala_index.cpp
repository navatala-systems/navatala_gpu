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

// Navatala C++ Wrapper Layer - Index Implementation
// Part of the Navatala GPU Runtime Library
#include "../include/navatala/index.hpp"
#include "../include/navatala/kmeans.hpp"
#include "../include/navatala/navatala_ffi.h"

#include <cstring>
#include <fstream>
#include <span>

namespace navatala {

// ============================================================================
// CAGRA Index Implementation
// ============================================================================

cagra_index::cagra_index() noexcept
    : handle_(nullptr)
    , parent_resources_(nullptr)
    , dims_(0)
    , n_vectors_(0)
    , graph_degree_(0)
    , metric_(metric_type::l2)
{}

cagra_index::cagra_index(NavatalaVectorIndex* handle, resources* res,
                         size_t dims, size_t n_vectors, size_t graph_degree,
                         metric_type metric)
    : handle_(handle)
    , parent_resources_(res)
    , dims_(dims)
    , n_vectors_(n_vectors)
    , graph_degree_(graph_degree)
    , metric_(metric)
{}

cagra_index::~cagra_index() noexcept {
    if (handle_) {
        navatala_vector_index_destroy(handle_);
        handle_ = nullptr;
    }
}

cagra_index::cagra_index(cagra_index&& other) noexcept
    : handle_(other.handle_)
    , parent_resources_(other.parent_resources_)
    , dims_(other.dims_)
    , n_vectors_(other.n_vectors_)
    , graph_degree_(other.graph_degree_)
    , metric_(other.metric_)
{
    other.handle_ = nullptr;
    other.parent_resources_ = nullptr;
    other.dims_ = 0;
    other.n_vectors_ = 0;
    other.graph_degree_ = 0;
}

cagra_index& cagra_index::operator=(cagra_index&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            navatala_vector_index_destroy(handle_);
        }
        handle_ = other.handle_;
        parent_resources_ = other.parent_resources_;
        dims_ = other.dims_;
        n_vectors_ = other.n_vectors_;
        graph_degree_ = other.graph_degree_;
        metric_ = other.metric_;

        other.handle_ = nullptr;
        other.parent_resources_ = nullptr;
        other.dims_ = 0;
        other.n_vectors_ = 0;
        other.graph_degree_ = 0;
    }
    return *this;
}

result<cagra_index> cagra_index::build(
    resources& res,
    const dataset& data,
    const cagra_build_params& params)
{
    if (!data.is_valid()) {
        return result<cagra_index>(error_code::invalid_handle, "dataset is invalid");
    }

    NavatalaCagraBuildParams ffi_params;
    ffi_params.graph_degree = params.graph_degree;
    ffi_params.intermediate_graph_degree = params.intermediate_graph_degree;
    ffi_params.build_algo = params.build_algo;
    ffi_params.nn_descent_niter = params.nn_descent_niter;
    ffi_params.metric = static_cast<int>(params.metric);

    NavatalaVectorIndex* handle = nullptr;
    NavatalaErrorCode err = navatala_cagra_build(
        res.handle(),
        data.device_ptr(),
        data.n_vectors(),
        data.dims(),
        &ffi_params,
        &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<cagra_index>(translate_error(err), "failed to build CAGRA index");
    }

    return result<cagra_index>(cagra_index(
        handle, &res, data.dims(), data.n_vectors(), params.graph_degree, params.metric));
}

result<cagra_index> cagra_index::load(resources& res, const std::string& path) {
    NavatalaVectorIndex* handle = nullptr;
    size_t dims = 0, n_vectors = 0, graph_degree = 0;
    int metric = 0;

    NavatalaErrorCode err = navatala_cagra_load(
        res.handle(), path.c_str(), &handle, &dims, &n_vectors, &graph_degree, &metric);

    if (err != NAVATALA_SUCCESS) {
        return result<cagra_index>(translate_error(err), "failed to load CAGRA index");
    }

    return result<cagra_index>(cagra_index(
        handle, &res, dims, n_vectors, graph_degree, static_cast<metric_type>(metric)));
}

result<search_results> cagra_index::search(
    const dataset& queries,
    size_t k,
    const cagra_search_params& params,
    resources& res) const
{
    if (!is_valid()) {
        return result<search_results>(error_code::invalid_handle, "index is invalid");
    }

    if (!queries.is_valid()) {
        return result<search_results>(error_code::invalid_handle, "queries dataset is invalid");
    }

    if (queries.dims() != dims_) {
        return result<search_results>(error_code::invalid_param,
                                       "query dimensions do not match index");
    }

    // Create results container
    auto results = search_results::create(res, queries.n_vectors(), k);
    if (!results.has_value()) {
        return result<search_results>(results.error(), "failed to create results");
    }

    NavatalaCagraSearchParams ffi_params;
    ffi_params.max_iterations = params.max_iterations;
    ffi_params.itopk_size = params.itopk_size;
    ffi_params.search_width = params.search_width;
    ffi_params.min_iterations = params.min_iterations;
    ffi_params.thread_block_size = params.thread_block_size;
    ffi_params.hashmap_mode = params.hashmap_mode;
    ffi_params.hashmap_max_fill_rate = params.hashmap_max_fill_rate;
    ffi_params.num_random_samplings = params.num_random_samplings;
    ffi_params.rand_xor_mask = params.rand_xor_mask;

    NavatalaErrorCode err = navatala_cagra_search(
        handle_,
        queries.device_ptr(),
        queries.n_vectors(),
        k,
        &ffi_params,
        results.value().indices_buffer().data(),
        results.value().distances_buffer().data(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<search_results>(translate_error(err), "CAGRA search failed");
    }

    return results;
}

status cagra_index::save(const std::string& path) const {
    if (!is_valid()) {
        return status(error_code::invalid_handle, "index is invalid");
    }

    NavatalaErrorCode err = navatala_cagra_save(handle_, path.c_str());
    if (err != NAVATALA_SUCCESS) {
        return status(translate_error(err), "failed to save CAGRA index");
    }

    return status();
}

// ============================================================================
// IVF-PQ Index Implementation
// ============================================================================

ivfpq_index::ivfpq_index() noexcept
    : handle_(nullptr)
    , parent_resources_(nullptr)
    , dims_(0)
    , n_vectors_(0)
    , n_lists_(0)
    , pq_dim_(0)
    , pq_bits_(0)
    , metric_(metric_type::l2)
{}

ivfpq_index::ivfpq_index(NavatalaVectorIndex* handle, resources* res,
                         size_t dims, size_t n_vectors, size_t n_lists,
                         size_t pq_dim, size_t pq_bits, metric_type metric)
    : handle_(handle)
    , parent_resources_(res)
    , dims_(dims)
    , n_vectors_(n_vectors)
    , n_lists_(n_lists)
    , pq_dim_(pq_dim)
    , pq_bits_(pq_bits)
    , metric_(metric)
{}

ivfpq_index::~ivfpq_index() noexcept {
    if (handle_) {
        navatala_vector_index_destroy(handle_);
        handle_ = nullptr;
    }
}

ivfpq_index::ivfpq_index(ivfpq_index&& other) noexcept
    : handle_(other.handle_)
    , parent_resources_(other.parent_resources_)
    , dims_(other.dims_)
    , n_vectors_(other.n_vectors_)
    , n_lists_(other.n_lists_)
    , pq_dim_(other.pq_dim_)
    , pq_bits_(other.pq_bits_)
    , metric_(other.metric_)
{
    other.handle_ = nullptr;
    other.parent_resources_ = nullptr;
    other.dims_ = 0;
    other.n_vectors_ = 0;
    other.n_lists_ = 0;
    other.pq_dim_ = 0;
    other.pq_bits_ = 0;
}

ivfpq_index& ivfpq_index::operator=(ivfpq_index&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            navatala_vector_index_destroy(handle_);
        }
        handle_ = other.handle_;
        parent_resources_ = other.parent_resources_;
        dims_ = other.dims_;
        n_vectors_ = other.n_vectors_;
        n_lists_ = other.n_lists_;
        pq_dim_ = other.pq_dim_;
        pq_bits_ = other.pq_bits_;
        metric_ = other.metric_;

        other.handle_ = nullptr;
        other.parent_resources_ = nullptr;
        other.dims_ = 0;
        other.n_vectors_ = 0;
        other.n_lists_ = 0;
        other.pq_dim_ = 0;
        other.pq_bits_ = 0;
    }
    return *this;
}

result<ivfpq_index> ivfpq_index::build(
    resources& res,
    const dataset& data,
    const ivfpq_build_params& params)
{
    if (!data.is_valid()) {
        return result<ivfpq_index>(error_code::invalid_handle, "dataset is invalid");
    }

    NavatalaIvfpqBuildParams ffi_params;
    ffi_params.n_lists = params.n_lists;
    ffi_params.pq_dim = params.pq_dim == 0 ? data.dims() / 4 : params.pq_dim;
    ffi_params.pq_bits = params.pq_bits;
    ffi_params.kmeans_n_iters = params.kmeans_n_iters;
    ffi_params.kmeans_trainset_fraction = params.kmeans_trainset_fraction;
    ffi_params.metric = static_cast<int>(params.metric);

    NavatalaVectorIndex* handle = nullptr;
    NavatalaErrorCode err = navatala_ivfpq_build(
        res.handle(),
        data.device_ptr(),
        data.n_vectors(),
        data.dims(),
        &ffi_params,
        &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<ivfpq_index>(translate_error(err), "failed to build IVF-PQ index");
    }

    return result<ivfpq_index>(ivfpq_index(
        handle, &res, data.dims(), data.n_vectors(), params.n_lists,
        ffi_params.pq_dim, params.pq_bits, params.metric));
}

result<ivfpq_index> ivfpq_index::load(resources& res, const std::string& path) {
    NavatalaVectorIndex* handle = nullptr;
    size_t dims = 0, n_vectors = 0, n_lists = 0, pq_dim = 0, pq_bits = 0;
    int metric = 0;

    NavatalaErrorCode err = navatala_ivfpq_load(
        res.handle(), path.c_str(), &handle, &dims, &n_vectors,
        &n_lists, &pq_dim, &pq_bits, &metric);

    if (err != NAVATALA_SUCCESS) {
        return result<ivfpq_index>(translate_error(err), "failed to load IVF-PQ index");
    }

    return result<ivfpq_index>(ivfpq_index(
        handle, &res, dims, n_vectors, n_lists, pq_dim, pq_bits,
        static_cast<metric_type>(metric)));
}

result<search_results> ivfpq_index::search(
    const dataset& queries,
    size_t k,
    const ivfpq_search_params& params,
    resources& res) const
{
    if (!is_valid()) {
        return result<search_results>(error_code::invalid_handle, "index is invalid");
    }

    if (!queries.is_valid()) {
        return result<search_results>(error_code::invalid_handle, "queries dataset is invalid");
    }

    if (queries.dims() != dims_) {
        return result<search_results>(error_code::invalid_param,
                                       "query dimensions do not match index");
    }

    auto results = search_results::create(res, queries.n_vectors(), k);
    if (!results.has_value()) {
        return result<search_results>(results.error(), "failed to create results");
    }

    NavatalaIvfpqSearchParams ffi_params;
    ffi_params.n_probes = params.n_probes;
    ffi_params.lut_dtype = params.lut_dtype;
    ffi_params.internal_distance_dtype = params.internal_distance_dtype;
    ffi_params.preferred_shmem_carveout = params.preferred_shmem_carveout;

    NavatalaErrorCode err = navatala_ivfpq_search(
        handle_,
        queries.device_ptr(),
        queries.n_vectors(),
        k,
        &ffi_params,
        results.value().indices_buffer().data(),
        results.value().distances_buffer().data(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<search_results>(translate_error(err), "IVF-PQ search failed");
    }

    return results;
}

status ivfpq_index::save(const std::string& path) const {
    if (!is_valid()) {
        return status(error_code::invalid_handle, "index is invalid");
    }

    NavatalaErrorCode err = navatala_ivfpq_save(handle_, path.c_str());
    if (err != NAVATALA_SUCCESS) {
        return status(translate_error(err), "failed to save IVF-PQ index");
    }

    return status();
}

// ============================================================================
// IVF-Flat Index Implementation
// ============================================================================

ivfflat_index::ivfflat_index() noexcept
    : handle_(nullptr)
    , parent_resources_(nullptr)
    , dims_(0)
    , n_vectors_(0)
    , n_lists_(0)
    , metric_(metric_type::l2)
{}

ivfflat_index::ivfflat_index(NavatalaVectorIndex* handle, resources* res,
                             size_t dims, size_t n_vectors, size_t n_lists,
                             metric_type metric)
    : handle_(handle)
    , parent_resources_(res)
    , dims_(dims)
    , n_vectors_(n_vectors)
    , n_lists_(n_lists)
    , metric_(metric)
{}

ivfflat_index::~ivfflat_index() noexcept {
    if (handle_) {
        navatala_vector_index_destroy(handle_);
        handle_ = nullptr;
    }
}

ivfflat_index::ivfflat_index(ivfflat_index&& other) noexcept
    : handle_(other.handle_)
    , parent_resources_(other.parent_resources_)
    , dims_(other.dims_)
    , n_vectors_(other.n_vectors_)
    , n_lists_(other.n_lists_)
    , metric_(other.metric_)
{
    other.handle_ = nullptr;
    other.parent_resources_ = nullptr;
    other.dims_ = 0;
    other.n_vectors_ = 0;
    other.n_lists_ = 0;
}

ivfflat_index& ivfflat_index::operator=(ivfflat_index&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            navatala_vector_index_destroy(handle_);
        }
        handle_ = other.handle_;
        parent_resources_ = other.parent_resources_;
        dims_ = other.dims_;
        n_vectors_ = other.n_vectors_;
        n_lists_ = other.n_lists_;
        metric_ = other.metric_;

        other.handle_ = nullptr;
        other.parent_resources_ = nullptr;
        other.dims_ = 0;
        other.n_vectors_ = 0;
        other.n_lists_ = 0;
    }
    return *this;
}

result<ivfflat_index> ivfflat_index::build(
    resources& res,
    const dataset& data,
    const ivfflat_build_params& params)
{
    if (!data.is_valid()) {
        return result<ivfflat_index>(error_code::invalid_handle, "dataset is invalid");
    }

    NavatalaIvfflatBuildParams ffi_params;
    ffi_params.n_lists = params.n_lists;
    ffi_params.kmeans_n_iters = params.kmeans_n_iters;
    ffi_params.kmeans_trainset_fraction = params.kmeans_trainset_fraction;
    ffi_params.metric = static_cast<int>(params.metric);

    NavatalaVectorIndex* handle = nullptr;
    NavatalaErrorCode err = navatala_ivfflat_build(
        res.handle(),
        data.device_ptr(),
        data.n_vectors(),
        data.dims(),
        &ffi_params,
        &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<ivfflat_index>(translate_error(err), "failed to build IVF-Flat index");
    }

    return result<ivfflat_index>(ivfflat_index(
        handle, &res, data.dims(), data.n_vectors(), params.n_lists, params.metric));
}

result<ivfflat_index> ivfflat_index::load(resources& res, const std::string& path) {
    NavatalaVectorIndex* handle = nullptr;
    size_t dims = 0, n_vectors = 0, n_lists = 0;
    int metric = 0;

    NavatalaErrorCode err = navatala_ivfflat_load(
        res.handle(), path.c_str(), &handle, &dims, &n_vectors, &n_lists, &metric);

    if (err != NAVATALA_SUCCESS) {
        return result<ivfflat_index>(translate_error(err), "failed to load IVF-Flat index");
    }

    return result<ivfflat_index>(ivfflat_index(
        handle, &res, dims, n_vectors, n_lists, static_cast<metric_type>(metric)));
}

result<search_results> ivfflat_index::search(
    const dataset& queries,
    size_t k,
    const ivfflat_search_params& params,
    resources& res) const
{
    if (!is_valid()) {
        return result<search_results>(error_code::invalid_handle, "index is invalid");
    }

    if (!queries.is_valid()) {
        return result<search_results>(error_code::invalid_handle, "queries dataset is invalid");
    }

    if (queries.dims() != dims_) {
        return result<search_results>(error_code::invalid_param,
                                       "query dimensions do not match index");
    }

    auto results = search_results::create(res, queries.n_vectors(), k);
    if (!results.has_value()) {
        return result<search_results>(results.error(), "failed to create results");
    }

    NavatalaIvfflatSearchParams ffi_params;
    ffi_params.n_probes = params.n_probes;

    NavatalaErrorCode err = navatala_ivfflat_search(
        handle_,
        queries.device_ptr(),
        queries.n_vectors(),
        k,
        &ffi_params,
        results.value().indices_buffer().data(),
        results.value().distances_buffer().data(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<search_results>(translate_error(err), "IVF-Flat search failed");
    }

    return results;
}

status ivfflat_index::save(const std::string& path) const {
    if (!is_valid()) {
        return status(error_code::invalid_handle, "index is invalid");
    }

    NavatalaErrorCode err = navatala_ivfflat_save(handle_, path.c_str());
    if (err != NAVATALA_SUCCESS) {
        return status(translate_error(err), "failed to save IVF-Flat index");
    }

    return status();
}

// ============================================================================
// Brute Force Index Implementation
// ============================================================================

brute_force_index::brute_force_index() noexcept
    : data_(nullptr)
    , parent_resources_(nullptr)
    , dims_(0)
    , n_vectors_(0)
    , metric_(metric_type::l2)
{}

brute_force_index::brute_force_index(const dataset* data, resources* res,
                                     size_t dims, size_t n_vectors, metric_type metric)
    : data_(data)
    , parent_resources_(res)
    , dims_(dims)
    , n_vectors_(n_vectors)
    , metric_(metric)
{}

brute_force_index::~brute_force_index() noexcept {
    // Non-owning reference, nothing to destroy
}

brute_force_index::brute_force_index(brute_force_index&& other) noexcept
    : data_(other.data_)
    , parent_resources_(other.parent_resources_)
    , dims_(other.dims_)
    , n_vectors_(other.n_vectors_)
    , metric_(other.metric_)
{
    other.data_ = nullptr;
    other.parent_resources_ = nullptr;
    other.dims_ = 0;
    other.n_vectors_ = 0;
}

brute_force_index& brute_force_index::operator=(brute_force_index&& other) noexcept {
    if (this != &other) {
        data_ = other.data_;
        parent_resources_ = other.parent_resources_;
        dims_ = other.dims_;
        n_vectors_ = other.n_vectors_;
        metric_ = other.metric_;

        other.data_ = nullptr;
        other.parent_resources_ = nullptr;
        other.dims_ = 0;
        other.n_vectors_ = 0;
    }
    return *this;
}

result<brute_force_index> brute_force_index::build(
    resources& res,
    const dataset& data,
    metric_type metric)
{
    if (!data.is_valid()) {
        return result<brute_force_index>(error_code::invalid_handle, "dataset is invalid");
    }

    // Brute force just stores a reference to the data
    return result<brute_force_index>(brute_force_index(
        &data, &res, data.dims(), data.n_vectors(), metric));
}

result<search_results> brute_force_index::search(
    const dataset& queries,
    size_t k,
    resources& res) const
{
    if (!is_valid()) {
        return result<search_results>(error_code::invalid_handle, "index is invalid");
    }

    if (!queries.is_valid()) {
        return result<search_results>(error_code::invalid_handle, "queries dataset is invalid");
    }

    if (queries.dims() != dims_) {
        return result<search_results>(error_code::invalid_param,
                                       "query dimensions do not match index");
    }

    auto results = search_results::create(res, queries.n_vectors(), k);
    if (!results.has_value()) {
        return result<search_results>(results.error(), "failed to create results");
    }

    NavatalaErrorCode err = navatala_brute_force_search(
        res.handle(),
        data_->device_ptr(),
        n_vectors_,
        dims_,
        queries.device_ptr(),
        queries.n_vectors(),
        k,
        static_cast<int>(metric_),
        results.value().indices_buffer().data(),
        results.value().distances_buffer().data(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<search_results>(translate_error(err), "brute force search failed");
    }

    return results;
}

// ============================================================================
// K-Means Implementation
// ============================================================================

result<kmeans_model> kmeans_model::fit(
    resources& res,
    const dataset& data,
    const kmeans_params& params)
{
    if (!data.is_valid()) {
        return result<kmeans_model>(error_code::invalid_handle, "dataset is invalid");
    }

    // Create centroids dataset
    auto centroids = dataset::create(res, params.n_clusters, data.dims(), data.dtype());
    if (!centroids.has_value()) {
        return result<kmeans_model>(centroids.error(), "failed to create centroids buffer");
    }

    kmeans_fit_result fit_result;

    NavatalaErrorCode err = navatala_kmeans_fit(
        res.handle(),
        data.device_ptr(),
        data.n_vectors(),
        data.dims(),
        params.n_clusters,
        params.max_iters,
        params.tol,
        params.init_method,
        params.seed,
        static_cast<int>(params.metric),
        centroids.value().device_ptr(),
        &fit_result.inertia,
        &fit_result.n_iter,
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<kmeans_model>(translate_error(err), "k-means fit failed");
    }

    fit_result.converged = fit_result.n_iter < params.max_iters;

    return result<kmeans_model>(kmeans_model(
        std::move(centroids.value()),
        params.n_clusters,
        data.dims(),
        params.metric,
        fit_result));
}

result<kmeans_model> kmeans_model::load(
    resources& res,
    const std::string& path)
{
    // Open file to check if it exists and read metadata
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return result<kmeans_model>(error_code::io_error, "failed to open file");
    }

    // Read header: n_clusters (8 bytes), dims (8 bytes)
    size_t n_clusters, dims;
    file.read(reinterpret_cast<char*>(&n_clusters), sizeof(n_clusters));
    file.read(reinterpret_cast<char*>(&dims), sizeof(dims));

    if (file.fail()) {
        return result<kmeans_model>(error_code::io_error, "failed to read header");
    }

    // Read centroids
    std::vector<float> centroid_data(n_clusters * dims);
    file.read(reinterpret_cast<char*>(centroid_data.data()), n_clusters * dims * sizeof(float));

    if (file.fail()) {
        return result<kmeans_model>(error_code::io_error, "failed to read centroids");
    }

    // Create dataset from centroids
    auto centroids = dataset::create(
        res, std::span<const float>(centroid_data), n_clusters, dims, data_type::float32);

    if (!centroids.has_value()) {
        return result<kmeans_model>(centroids.error(), "failed to create centroids dataset");
    }

    return result<kmeans_model>(kmeans_model(
        std::move(centroids.value()),
        n_clusters,
        dims,
        metric_type::l2,
        std::nullopt));
}

result<buffer> kmeans_model::predict(
    const dataset& data,
    resources& res) const
{
    if (!is_valid()) {
        return result<buffer>(error_code::invalid_handle, "model is invalid");
    }

    if (data.dims() != dims_) {
        return result<buffer>(error_code::invalid_param, "dimension mismatch");
    }

    auto labels = buffer::create(res, {data.n_vectors()}, data_type::int32);
    if (!labels.has_value()) {
        return labels;
    }

    NavatalaErrorCode err = navatala_kmeans_predict(
        res.handle(),
        centroids_.device_ptr(),
        n_clusters_,
        dims_,
        data.device_ptr(),
        data.n_vectors(),
        static_cast<int>(metric_),
        labels.value().data(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "k-means predict failed");
    }

    return labels;
}

result<buffer> kmeans_model::transform(
    const dataset& data,
    resources& res) const
{
    if (!is_valid()) {
        return result<buffer>(error_code::invalid_handle, "model is invalid");
    }

    if (data.dims() != dims_) {
        return result<buffer>(error_code::invalid_param, "dimension mismatch");
    }

    auto distances = buffer::create(res, {data.n_vectors(), n_clusters_}, data_type::float32);
    if (!distances.has_value()) {
        return distances;
    }

    NavatalaErrorCode err = navatala_kmeans_transform(
        res.handle(),
        centroids_.device_ptr(),
        n_clusters_,
        dims_,
        data.device_ptr(),
        data.n_vectors(),
        static_cast<int>(metric_),
        distances.value().data(),
        res.default_queue().handle());

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "k-means transform failed");
    }

    return distances;
}

result<std::pair<buffer, buffer>> kmeans_model::predict_transform(
    const dataset& data,
    resources& res) const
{
    auto labels = predict(data, res);
    if (!labels.has_value()) {
        return result<std::pair<buffer, buffer>>(labels.error(), "predict failed");
    }

    auto distances = transform(data, res);
    if (!distances.has_value()) {
        return result<std::pair<buffer, buffer>>(distances.error(), "transform failed");
    }

    return result<std::pair<buffer, buffer>>(
        std::make_pair(std::move(labels.value()), std::move(distances.value())));
}

status kmeans_model::save(const std::string& path) const
{
    if (!is_valid()) {
        return status(error_code::invalid_handle, "model is invalid");
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return status(error_code::io_error, "failed to open file for writing");
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&n_clusters_), sizeof(n_clusters_));
    file.write(reinterpret_cast<const char*>(&dims_), sizeof(dims_));

    // This would need a host copy of centroids - simplified for now
    // In real implementation, we'd copy centroids to host first

    if (file.fail()) {
        return status(error_code::io_error, "failed to write model");
    }

    return status();
}

} // namespace navatala
