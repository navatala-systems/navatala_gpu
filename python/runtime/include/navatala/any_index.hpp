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

// Navatala C++ Wrapper Layer - Type-Erased Index
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"
#include "error.hpp"
#include "index.hpp"

#include <variant>
#include <memory>

/// @file navatala/any_index.hpp
/// @brief Type-erased container for any vector search index.
///
/// This header provides:
/// - any_index: A std::variant-based container that can hold any index type
/// - Unified search interface across all index types

namespace navatala {

// ============================================================================
// Any Index Class
// ============================================================================

/// @brief Type-erased container for any vector search index.
///
/// Uses std::variant internally to hold any of the supported index types.
/// Provides a unified interface for common operations like search.
///
/// Example:
/// @code
///   auto res = resources::create().value();
///   auto data = dataset::create(res, ...).value();
///
///   // Build any type of index
///   any_index idx = cagra_index::build(res, data).value();
///
///   // Or switch based on configuration
///   if (use_ivfpq) {
///       idx = ivfpq_index::build(res, data).value();
///   }
///
///   // Search works uniformly
///   auto results = idx.search(queries, k, res);
/// @endcode
class any_index {
public:
    /// The variant type holding all possible index types.
    using variant_type = std::variant<
        std::monostate,       // Empty/invalid state
        cagra_index,
        ivfpq_index,
        ivfflat_index,
        brute_force_index
    >;

    // -------------------- Constructors --------------------

    /// Default constructor creates an empty (invalid) index.
    any_index() noexcept : index_(std::monostate{}) {}

    /// Construct from a CAGRA index.
    any_index(cagra_index&& idx) : index_(std::move(idx)) {}

    /// Construct from an IVF-PQ index.
    any_index(ivfpq_index&& idx) : index_(std::move(idx)) {}

    /// Construct from an IVF-Flat index.
    any_index(ivfflat_index&& idx) : index_(std::move(idx)) {}

    /// Construct from a brute-force index.
    any_index(brute_force_index&& idx) : index_(std::move(idx)) {}

    // -------------------- Move Semantics --------------------

    any_index(any_index&& other) noexcept = default;
    any_index& operator=(any_index&& other) noexcept = default;

    // Non-copyable (indices are move-only)
    any_index(const any_index&) = delete;
    any_index& operator=(const any_index&) = delete;

    // -------------------- Accessors --------------------

    /// Returns true if the index is valid (not empty).
    [[nodiscard]] bool is_valid() const noexcept {
        return !std::holds_alternative<std::monostate>(index_);
    }

    /// Explicit bool conversion.
    explicit operator bool() const noexcept { return is_valid(); }

    /// Returns the index type.
    [[nodiscard]] knn_index_type type() const noexcept {
        return std::visit([](const auto& idx) -> knn_index_type {
            using T = std::decay_t<decltype(idx)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return knn_index_type::brute_force;  // Fallback for empty
            } else {
                return idx.type();
            }
        }, index_);
    }

    /// Returns the dimensionality.
    [[nodiscard]] size_t dims() const noexcept {
        return std::visit([](const auto& idx) -> size_t {
            using T = std::decay_t<decltype(idx)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return 0;
            } else {
                return idx.dims();
            }
        }, index_);
    }

    /// Returns the number of vectors.
    [[nodiscard]] size_t size() const noexcept {
        return std::visit([](const auto& idx) -> size_t {
            using T = std::decay_t<decltype(idx)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return 0;
            } else {
                return idx.size();
            }
        }, index_);
    }

    /// Returns the distance metric.
    [[nodiscard]] metric_type metric() const noexcept {
        return std::visit([](const auto& idx) -> metric_type {
            using T = std::decay_t<decltype(idx)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return metric_type::l2;
            } else {
                return idx.metric();
            }
        }, index_);
    }

    // -------------------- Type Access --------------------

    /// Returns true if the index is a CAGRA index.
    [[nodiscard]] bool is_cagra() const noexcept {
        return std::holds_alternative<cagra_index>(index_);
    }

    /// Returns true if the index is an IVF-PQ index.
    [[nodiscard]] bool is_ivfpq() const noexcept {
        return std::holds_alternative<ivfpq_index>(index_);
    }

    /// Returns true if the index is an IVF-Flat index.
    [[nodiscard]] bool is_ivfflat() const noexcept {
        return std::holds_alternative<ivfflat_index>(index_);
    }

    /// Returns true if the index is a brute-force index.
    [[nodiscard]] bool is_brute_force() const noexcept {
        return std::holds_alternative<brute_force_index>(index_);
    }

    /// Gets the underlying CAGRA index (throws if wrong type).
    [[nodiscard]] cagra_index& as_cagra() {
        return std::get<cagra_index>(index_);
    }

    [[nodiscard]] const cagra_index& as_cagra() const {
        return std::get<cagra_index>(index_);
    }

    /// Gets the underlying IVF-PQ index.
    [[nodiscard]] ivfpq_index& as_ivfpq() {
        return std::get<ivfpq_index>(index_);
    }

    [[nodiscard]] const ivfpq_index& as_ivfpq() const {
        return std::get<ivfpq_index>(index_);
    }

    /// Gets the underlying IVF-Flat index.
    [[nodiscard]] ivfflat_index& as_ivfflat() {
        return std::get<ivfflat_index>(index_);
    }

    [[nodiscard]] const ivfflat_index& as_ivfflat() const {
        return std::get<ivfflat_index>(index_);
    }

    /// Gets the underlying brute-force index.
    [[nodiscard]] brute_force_index& as_brute_force() {
        return std::get<brute_force_index>(index_);
    }

    [[nodiscard]] const brute_force_index& as_brute_force() const {
        return std::get<brute_force_index>(index_);
    }

    /// Returns the underlying variant for advanced use.
    [[nodiscard]] const variant_type& variant() const noexcept { return index_; }
    [[nodiscard]] variant_type& variant() noexcept { return index_; }

    // -------------------- Search --------------------

    /// Performs k-NN search using the appropriate index type.
    /// @param queries Query vectors.
    /// @param k Number of neighbors.
    /// @param res GPU resources.
    /// @return An expected<search_results> or an error.
    [[nodiscard]] result<search_results> search(
        const dataset& queries,
        size_t k,
        resources& res) const
    {
        return std::visit([&](const auto& idx) -> result<search_results> {
            using T = std::decay_t<decltype(idx)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return result<search_results>(error_code::invalid_handle, "index is empty");
            } else {
                return idx.search(queries, k, res);
            }
        }, index_);
    }

private:
    variant_type index_;
};

// ============================================================================
// Convenience Factory Functions
// ============================================================================

/// @brief Builds an any_index of the specified type.
/// @param type The index type to build.
/// @param res GPU resources.
/// @param data Dataset to index.
/// @param metric Distance metric.
/// @return An expected<any_index> or an error.
[[nodiscard]] inline result<any_index> build_index(
    knn_index_type type,
    resources& res,
    const dataset& data,
    metric_type metric = metric_type::l2)
{
    switch (type) {
        case knn_index_type::cagra: {
            cagra_build_params params;
            params.metric = metric;
            auto idx = cagra_index::build(res, data, params);
            if (!idx.has_value()) {
                return result<any_index>(idx.error(), "failed to build CAGRA index");
            }
            return result<any_index>(any_index(std::move(idx.value())));
        }

        case knn_index_type::ivfpq: {
            ivfpq_build_params params;
            params.metric = metric;
            auto idx = ivfpq_index::build(res, data, params);
            if (!idx.has_value()) {
                return result<any_index>(idx.error(), "failed to build IVF-PQ index");
            }
            return result<any_index>(any_index(std::move(idx.value())));
        }

        case knn_index_type::ivfflat: {
            ivfflat_build_params params;
            params.metric = metric;
            auto idx = ivfflat_index::build(res, data, params);
            if (!idx.has_value()) {
                return result<any_index>(idx.error(), "failed to build IVF-Flat index");
            }
            return result<any_index>(any_index(std::move(idx.value())));
        }

        case knn_index_type::brute_force: {
            auto idx = brute_force_index::build(res, data, metric);
            if (!idx.has_value()) {
                return result<any_index>(idx.error(), "failed to build brute-force index");
            }
            return result<any_index>(any_index(std::move(idx.value())));
        }

        default:
            return result<any_index>(error_code::invalid_param, "unknown index type");
    }
}

} // namespace navatala
