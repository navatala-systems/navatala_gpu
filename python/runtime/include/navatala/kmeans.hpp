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

// Navatala C++ Wrapper Layer - K-Means Clustering
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"
#include "error.hpp"
#include "types.hpp"
#include "buffer.hpp"
#include "dataset.hpp"
#include "resource.hpp"

#include <vector>
#include <optional>

/// @file navatala/kmeans.hpp
/// @brief GPU-accelerated k-means clustering.
///
/// This header provides:
/// - kmeans_params: Configuration for k-means training
/// - kmeans_model: Trained k-means model with centroids
/// - Predict and transform utilities

// Forward declare FFI types
struct NavatalaKMeansModel;

namespace navatala {

// ============================================================================
// K-Means Parameters
// ============================================================================

/// @brief Parameters for k-means clustering.
struct kmeans_params {
    /// Number of clusters.
    size_t n_clusters = 256;

    /// Maximum number of iterations.
    size_t max_iters = 300;

    /// Convergence tolerance (relative change in inertia).
    float tol = 1e-4f;

    /// Random seed for initialization.
    uint64_t seed = 42;

    /// Initialization method: 0 = random, 1 = k-means++.
    int init_method = 1;

    /// Number of runs with different initializations.
    int n_init = 1;

    /// Verbosity level (0 = silent, 1 = progress, 2 = detailed).
    int verbosity = 0;

    /// Distance metric (l2 or cosine).
    metric_type metric = metric_type::l2;
};

/// @brief Result of k-means fitting.
struct kmeans_fit_result {
    /// Final inertia (sum of squared distances to centroids).
    float inertia = 0.0f;

    /// Number of iterations performed.
    size_t n_iter = 0;

    /// Whether the algorithm converged.
    bool converged = false;
};

// ============================================================================
// K-Means Model
// ============================================================================

/// @brief GPU-accelerated k-means clustering model.
///
/// Performs k-means clustering on GPU and provides methods for
/// predicting cluster assignments and transforming data.
///
/// Example:
/// @code
///   auto res = resources::create().value();
///   auto data = dataset::create(res, vectors, n, dims, data_type::float32).value();
///
///   // Configure k-means
///   kmeans_params params;
///   params.n_clusters = 100;
///   params.max_iters = 100;
///
///   // Fit model
///   auto model = kmeans_model::fit(res, data, params).value();
///
///   // Get cluster assignments
///   auto labels = model.predict(data, res).value();
///
///   // Get centroids
///   auto centroids = model.centroids();
/// @endcode
class kmeans_model {
public:
    // -------------------- Factory Methods --------------------

    /// Fits a k-means model to the data.
    /// @param res GPU resources.
    /// @param data Training data.
    /// @param params K-means parameters.
    /// @return An expected<kmeans_model> or an error.
    [[nodiscard]] static result<kmeans_model> fit(
        resources& res,
        const dataset& data,
        const kmeans_params& params = {});

    /// Creates a model from existing centroids.
    /// @param res GPU resources.
    /// @param centroids Pre-computed centroids [n_clusters, dims].
    /// @return An expected<kmeans_model> or an error.
    [[nodiscard]] static result<kmeans_model> from_centroids(
        resources& res,
        dataset&& centroids);

    /// Loads a model from file.
    /// @param res GPU resources.
    /// @param path Path to the model file.
    /// @return An expected<kmeans_model> or an error.
    [[nodiscard]] static result<kmeans_model> load(
        resources& res,
        const std::string& path);

    // -------------------- Constructors/Destructor --------------------

    /// Default constructor creates an invalid model.
    kmeans_model() noexcept;

    /// Destructor.
    ~kmeans_model() noexcept;

    // Non-copyable
    kmeans_model(const kmeans_model&) = delete;
    kmeans_model& operator=(const kmeans_model&) = delete;

    // Movable
    kmeans_model(kmeans_model&& other) noexcept;
    kmeans_model& operator=(kmeans_model&& other) noexcept;

    // -------------------- Accessors --------------------

    /// Returns the number of clusters.
    [[nodiscard]] size_t n_clusters() const noexcept { return n_clusters_; }

    /// Returns the dimensionality.
    [[nodiscard]] size_t dims() const noexcept { return dims_; }

    /// Returns the distance metric.
    [[nodiscard]] metric_type metric() const noexcept { return metric_; }

    /// Returns true if the model is valid.
    [[nodiscard]] bool is_valid() const noexcept { return centroids_.is_valid(); }

    /// Explicit bool conversion.
    explicit operator bool() const noexcept { return is_valid(); }

    /// Returns the fit result (if available).
    [[nodiscard]] const std::optional<kmeans_fit_result>& fit_result() const noexcept {
        return fit_result_;
    }

    /// Returns the centroids dataset.
    [[nodiscard]] const dataset& centroids() const noexcept { return centroids_; }

    /// Returns centroids as host vectors.
    [[nodiscard]] std::vector<float> centroids_to_host(resources& res) const {
        return centroids_.to_host<float>(res);
    }

    // -------------------- Prediction --------------------

    /// Predicts cluster labels for new data.
    /// @param data Data to predict [n_samples, dims].
    /// @param res GPU resources.
    /// @return An expected<buffer> containing labels [n_samples] or an error.
    [[nodiscard]] result<buffer> predict(
        const dataset& data,
        resources& res) const;

    /// Predicts labels and returns them as host vector.
    [[nodiscard]] result<std::vector<int32_t>> predict_to_host(
        const dataset& data,
        resources& res) const
    {
        auto labels = predict(data, res);
        if (!labels.has_value()) {
            return result<std::vector<int32_t>>(labels.error(), "prediction failed");
        }
        return result<std::vector<int32_t>>(labels.value().to_host<int32_t>(res));
    }

    /// Computes distances to cluster centroids.
    /// @param data Data to transform [n_samples, dims].
    /// @param res GPU resources.
    /// @return An expected<buffer> with distances [n_samples, n_clusters] or error.
    [[nodiscard]] result<buffer> transform(
        const dataset& data,
        resources& res) const;

    /// Predict labels and compute distances in one call.
    /// @param data Data to process.
    /// @param res GPU resources.
    /// @return Tuple of (labels, distances) buffers.
    [[nodiscard]] result<std::pair<buffer, buffer>> predict_transform(
        const dataset& data,
        resources& res) const;

    // -------------------- Persistence --------------------

    /// Saves the model to file.
    [[nodiscard]] status save(const std::string& path) const;

private:
    // Private constructor
    kmeans_model(dataset centroids, size_t n_clusters, size_t dims,
                 metric_type metric, std::optional<kmeans_fit_result> fit_result);

    dataset centroids_;
    size_t n_clusters_;
    size_t dims_;
    metric_type metric_;
    std::optional<kmeans_fit_result> fit_result_;
    resources* parent_resources_;
};

// ============================================================================
// Inline Implementations
// ============================================================================

inline kmeans_model::kmeans_model() noexcept
    : centroids_()
    , n_clusters_(0)
    , dims_(0)
    , metric_(metric_type::l2)
    , fit_result_(std::nullopt)
    , parent_resources_(nullptr)
{}

inline kmeans_model::kmeans_model(dataset centroids, size_t n_clusters, size_t dims,
                                   metric_type metric, std::optional<kmeans_fit_result> fit_result)
    : centroids_(std::move(centroids))
    , n_clusters_(n_clusters)
    , dims_(dims)
    , metric_(metric)
    , fit_result_(fit_result)
    , parent_resources_(nullptr)
{}

inline kmeans_model::~kmeans_model() noexcept = default;

inline kmeans_model::kmeans_model(kmeans_model&& other) noexcept
    : centroids_(std::move(other.centroids_))
    , n_clusters_(other.n_clusters_)
    , dims_(other.dims_)
    , metric_(other.metric_)
    , fit_result_(std::move(other.fit_result_))
    , parent_resources_(other.parent_resources_)
{
    other.n_clusters_ = 0;
    other.dims_ = 0;
    other.parent_resources_ = nullptr;
}

inline kmeans_model& kmeans_model::operator=(kmeans_model&& other) noexcept {
    if (this != &other) {
        centroids_ = std::move(other.centroids_);
        n_clusters_ = other.n_clusters_;
        dims_ = other.dims_;
        metric_ = other.metric_;
        fit_result_ = std::move(other.fit_result_);
        parent_resources_ = other.parent_resources_;

        other.n_clusters_ = 0;
        other.dims_ = 0;
        other.parent_resources_ = nullptr;
    }
    return *this;
}

inline result<kmeans_model> kmeans_model::from_centroids(
    resources& res,
    dataset&& centroids)
{
    if (!centroids.is_valid()) {
        return result<kmeans_model>(error_code::invalid_handle, "centroids invalid");
    }

    size_t n_clusters = centroids.n_vectors();
    size_t dims = centroids.dims();

    return result<kmeans_model>(kmeans_model(
        std::move(centroids), n_clusters, dims, metric_type::l2, std::nullopt));
}

} // namespace navatala
