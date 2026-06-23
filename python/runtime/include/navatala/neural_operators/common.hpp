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

// Navatala C++ Wrapper Layer - Neural Operators Common Types
// Part of the Navatala GPU Runtime Library
//
// WARNING: This module is EXPERIMENTAL. The API may change without notice.

#pragma once

#include "../buffer.hpp"
#include "../types.hpp"
#include "../error.hpp"
#include "../queue.hpp"
#include "../resource.hpp"

#include <vector>
#include <string>
#include <string_view>
#include <concepts>

/// @file navatala/neural_operators/common.hpp
/// @brief Common types and concepts for PhysicsNeMo neural operators.
///
/// This header provides:
/// - Activation function and padding enumerations
/// - Forward declarations for neural operator configurations
/// - NeuralOperatorLayer concept for generic programming
///
/// @warning This module is EXPERIMENTAL. The API may change without notice.

namespace navatala {
namespace neural_operators {

// ============================================================================
// Activation Functions
// ============================================================================

/// @brief Common activation functions used in neural networks.
enum class activation : uint32_t {
    relu = 0,       ///< Rectified Linear Unit: max(0, x)
    gelu = 1,       ///< Gaussian Error Linear Unit
    sigmoid = 2,    ///< Sigmoid: 1 / (1 + exp(-x))
    tanh = 3,       ///< Hyperbolic tangent
    swish = 4,      ///< Swish/SiLU: x * sigmoid(x)
    leaky_relu = 5, ///< Leaky ReLU: max(alpha * x, x)
    none = 255      ///< Linear activation (no non-linearity)
};

/// @brief Returns the name of an activation function.
[[nodiscard]] constexpr const char* activation_name(activation act) noexcept {
    switch (act) {
        case activation::relu:       return "relu";
        case activation::gelu:       return "gelu";
        case activation::sigmoid:    return "sigmoid";
        case activation::tanh:       return "tanh";
        case activation::swish:      return "swish";
        case activation::leaky_relu: return "leaky_relu";
        case activation::none:       return "none";
        default:                     return "unknown";
    }
}

// ============================================================================
// Padding Types
// ============================================================================

/// @brief Padding types for convolutional operations.
enum class padding : uint32_t {
    same = 0,   ///< Pad to maintain input dimensions
    valid = 1,  ///< No padding (output smaller than input)
    zeros = 2,  ///< Pad with zeros (explicit zero-padding)
    reflect = 3 ///< Reflect padding at boundaries
};

/// @brief Returns the name of a padding type.
[[nodiscard]] constexpr const char* padding_name(padding pad) noexcept {
    switch (pad) {
        case padding::same:    return "same";
        case padding::valid:   return "valid";
        case padding::zeros:   return "zeros";
        case padding::reflect: return "reflect";
        default:               return "unknown";
    }
}

// ============================================================================
// Forward Declarations
// ============================================================================

// Configuration structs (defined in their respective headers)
struct fno_config;
struct pino_config;
struct mgn_config;

// Layer classes (defined in their respective headers)
class fno_layer;
class pino_layer;
class mesh_graph_net;

// ============================================================================
// Neural Operator Concept
// ============================================================================

/// @brief Concept for a neural operator layer.
///
/// A type satisfying NeuralOperatorLayer must support:
/// - Forward inference: `forward(input, queue) -> result<buffer>`
/// - Model persistence: `save(path)` and `T::load(res, path)`
/// - Shape introspection: `input_shape()` and `output_shape()`
/// - Identification: `name()`
///
/// Example:
/// @code
///   template<NeuralOperatorLayer T>
///   void run_inference(T& model, const buffer& input, const queue& q) {
///       auto output = model.forward(input, q);
///       // ...
///   }
/// @endcode
template<typename T>
concept NeuralOperatorLayer = requires(T op, const buffer& input, const queue& q,
                                        const std::string& path, resources& res) {
    { op.forward(input, q) } -> std::same_as<result<buffer>>;
    { op.save(path) } -> std::same_as<void>;
    // Note: static load() cannot be checked in requires clause
    { op.input_shape() } -> std::same_as<const std::vector<size_t>&>;
    { op.output_shape() } -> std::same_as<const std::vector<size_t>&>;
    { op.name() } -> std::same_as<std::string_view>;
    { op.is_valid() } -> std::same_as<bool>;
};

// ============================================================================
// Weight Initialization
// ============================================================================

/// @brief Weight initialization methods for neural networks.
enum class weight_init : uint32_t {
    xavier_uniform = 0,  ///< Xavier/Glorot uniform initialization
    xavier_normal = 1,   ///< Xavier/Glorot normal initialization
    kaiming_uniform = 2, ///< Kaiming/He uniform initialization
    kaiming_normal = 3,  ///< Kaiming/He normal initialization
    zeros = 4,           ///< Initialize all weights to zero
    ones = 5,            ///< Initialize all weights to one
    random_uniform = 6,  ///< Uniform random in [-1, 1]
    random_normal = 7    ///< Standard normal initialization
};

/// @brief Returns the name of a weight initialization method.
[[nodiscard]] constexpr const char* weight_init_name(weight_init init) noexcept {
    switch (init) {
        case weight_init::xavier_uniform:  return "xavier_uniform";
        case weight_init::xavier_normal:   return "xavier_normal";
        case weight_init::kaiming_uniform: return "kaiming_uniform";
        case weight_init::kaiming_normal:  return "kaiming_normal";
        case weight_init::zeros:           return "zeros";
        case weight_init::ones:            return "ones";
        case weight_init::random_uniform:  return "random_uniform";
        case weight_init::random_normal:   return "random_normal";
        default:                           return "unknown";
    }
}

// ============================================================================
// Training Configuration
// ============================================================================

/// @brief Common training configuration shared across neural operators.
struct training_config {
    float learning_rate = 1e-3f;    ///< Learning rate for optimization
    float weight_decay = 0.0f;      ///< L2 regularization weight
    size_t batch_size = 32;         ///< Training batch size
    bool use_amp = false;           ///< Use automatic mixed precision
    uint64_t seed = 42;             ///< Random seed for reproducibility
};

// ============================================================================
// Spectral Configuration (shared by FNO/PINO)
// ============================================================================

/// @brief Configuration for spectral convolution layers.
struct spectral_config {
    size_t n_modes = 16;            ///< Number of Fourier modes to retain
    bool symmetric_modes = true;    ///< Use symmetric mode truncation
    data_type spectral_dtype = data_type::float32; ///< Base data type (complex uses 2x this)
};

// ============================================================================
// Utility Functions
// ============================================================================

/// @brief Computes the total number of elements from a shape vector.
[[nodiscard]] inline size_t shape_numel(const std::vector<size_t>& shape) noexcept {
    if (shape.empty()) return 0;
    size_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    return numel;
}

/// @brief Validates that two shapes are broadcastable.
[[nodiscard]] inline bool shapes_broadcastable(
    const std::vector<size_t>& a,
    const std::vector<size_t>& b) noexcept
{
    size_t min_dims = std::min(a.size(), b.size());
    for (size_t i = 0; i < min_dims; ++i) {
        size_t dim_a = a[a.size() - 1 - i];
        size_t dim_b = b[b.size() - 1 - i];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }
    return true;
}

} // namespace neural_operators
} // namespace navatala
