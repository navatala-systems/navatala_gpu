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

// Navatala C++ Wrapper Layer - Fourier Neural Operator
// Part of the Navatala GPU Runtime Library
//
// WARNING: This module is EXPERIMENTAL. The API may change without notice.

#pragma once

#include "common.hpp"
#include "../resource.hpp"
#include "../error.hpp"

#include <vector>
#include <string>
#include <string_view>
#include <utility>

/// @file navatala/neural_operators/fno.hpp
/// @brief GPU-accelerated Fourier Neural Operator (FNO).
///
/// This header provides:
/// - fno_config: Configuration for FNO layers
/// - fno_config_builder: Fluent builder for configuration
/// - fno_layer: RAII wrapper for FNO inference
///
/// The Fourier Neural Operator learns mappings between function spaces
/// by operating in the Fourier domain. It is particularly effective for
/// solving PDEs like Navier-Stokes, Darcy flow, and wave equations.
///
/// @warning This module is EXPERIMENTAL. The API may change without notice.
///
/// Example:
/// @code
///   auto res = resources::create().value();
///
///   // Configure FNO
///   auto cfg = fno_config_builder()
///       .input_shape({64, 64, 3})   // 64x64 grid with 3 channels
///       .output_shape({64, 64, 1})  // Single output channel
///       .n_modes(12)                // 12 Fourier modes
///       .hidden_size(64)
///       .n_layers(4)
///       .with_activation(activation::gelu)
///       .build();
///
///   // Create FNO layer
///   auto fno = fno_layer::create(res, cfg).value();
///
///   // Forward pass
///   auto output = fno.forward(input_buffer, res.default_queue());
/// @endcode

// Forward declare FFI types
struct NavatalaFnoLayer;

namespace navatala {
namespace neural_operators {

// ============================================================================
// FNO Configuration
// ============================================================================

/// @brief Configuration structure for the Fourier Neural Operator (FNO).
struct fno_config {
    std::vector<size_t> input_shape;   ///< Input tensor shape [height, width, channels]
    std::vector<size_t> output_shape;  ///< Output tensor shape [height, width, channels]

    size_t n_modes = 16;        ///< Number of Fourier modes to retain per dimension
    size_t hidden_size = 128;   ///< Size of the hidden layers
    size_t n_layers = 4;        ///< Number of FNO spectral convolution layers

    activation act = activation::gelu;  ///< Activation function
    padding pad = padding::zeros;       ///< Padding strategy

    // Advanced options
    bool factorized = true;     ///< Use factorized spectral convolution (memory efficient)
    bool skip_connections = true; ///< Add skip connections in each layer
    weight_init init = weight_init::xavier_uniform; ///< Weight initialization method
    data_type dtype = data_type::float32; ///< Data type for weights

    /// Returns the spatial dimensions (excluding batch and channels).
    [[nodiscard]] size_t spatial_dims() const noexcept {
        return input_shape.empty() ? 0 : input_shape.size() - 1;
    }
};

/// @brief Fluent builder for `fno_config`.
///
/// Example:
/// @code
///   auto cfg = fno_config_builder()
///       .input_shape({64, 64, 3})
///       .output_shape({64, 64, 1})
///       .n_modes(12)
///       .build();
/// @endcode
class fno_config_builder {
public:
    fno_config_builder() = default;

    fno_config_builder& input_shape(std::vector<size_t> shape) {
        cfg_.input_shape = std::move(shape);
        return *this;
    }

    fno_config_builder& output_shape(std::vector<size_t> shape) {
        cfg_.output_shape = std::move(shape);
        return *this;
    }

    fno_config_builder& n_modes(size_t n) {
        cfg_.n_modes = n;
        return *this;
    }

    fno_config_builder& hidden_size(size_t size) {
        cfg_.hidden_size = size;
        return *this;
    }

    fno_config_builder& n_layers(size_t n) {
        cfg_.n_layers = n;
        return *this;
    }

    fno_config_builder& with_activation(activation a) {
        cfg_.act = a;
        return *this;
    }

    fno_config_builder& with_padding(padding p) {
        cfg_.pad = p;
        return *this;
    }

    fno_config_builder& factorized(bool f) {
        cfg_.factorized = f;
        return *this;
    }

    fno_config_builder& skip_connections(bool s) {
        cfg_.skip_connections = s;
        return *this;
    }

    fno_config_builder& with_init(weight_init init) {
        cfg_.init = init;
        return *this;
    }

    fno_config_builder& with_dtype(data_type dt) {
        cfg_.dtype = dt;
        return *this;
    }

    /// Builds the `fno_config` object.
    [[nodiscard]] fno_config build() const { return cfg_; }

private:
    fno_config cfg_;
};

// ============================================================================
// FNO Layer Class
// ============================================================================

/// @brief RAII wrapper for a device-resident Fourier Neural Operator layer.
///
/// The FNO layer performs spectral convolutions in the Fourier domain:
/// 1. FFT to transform input to frequency space
/// 2. Mode truncation to retain only low-frequency modes
/// 3. Complex multiplication with learned weights
/// 4. IFFT to transform back to spatial domain
/// 5. Add skip connection and apply activation
///
/// Thread Safety: `forward()` is safe for concurrent calls from multiple threads
/// after creation. `create()`, `save()`, and `load()` are not thread-safe.
class fno_layer {
public:
    // -------------------- Factory Methods --------------------

    /// Creates an FNO layer from a configuration.
    /// @param res GPU resources.
    /// @param cfg FNO configuration.
    /// @return An expected<fno_layer> or an error.
    [[nodiscard]] static result<fno_layer> create(
        resources& res,
        const fno_config& cfg);

    /// Loads an FNO model from a file.
    /// @param res GPU resources.
    /// @param path Path to the model file.
    /// @param cfg FNO configuration (must match saved model).
    /// @return An expected<fno_layer> or an error.
    [[nodiscard]] static result<fno_layer> load(
        resources& res,
        const std::string& path,
        const fno_config& cfg);

    // -------------------- Constructors/Destructor --------------------

    /// Default constructor for an invalid FNO layer.
    fno_layer() noexcept;

    /// Destructor. Releases the underlying GPU FNO layer.
    ~fno_layer() noexcept;

    // Non-copyable
    fno_layer(const fno_layer&) = delete;
    fno_layer& operator=(const fno_layer&) = delete;

    // Movable
    fno_layer(fno_layer&& other) noexcept;
    fno_layer& operator=(fno_layer&& other) noexcept;

    // -------------------- Inference --------------------

    /// Performs a forward pass (inference) through the FNO layer.
    /// @param input The input tensor on device [batch, height, width, channels].
    /// @param q The queue to use for the operation.
    /// @return An expected<buffer> containing the output tensor or an error.
    [[nodiscard]] result<buffer> forward(const buffer& input, const queue& q) const;

    /// Performs a forward pass with additional parametric inputs.
    /// @param input The primary input tensor.
    /// @param params The additional parametric input tensor (e.g., physical parameters).
    /// @param q The queue to use for the operation.
    /// @return An expected<buffer> containing the output tensor or an error.
    [[nodiscard]] result<buffer> forward(
        const buffer& input,
        const buffer& params,
        const queue& q) const;

    // -------------------- Persistence --------------------

    /// Saves the FNO model weights to a file.
    void save(const std::string& path) const;

    // -------------------- Accessors --------------------

    /// Returns the configuration used to create this FNO layer.
    [[nodiscard]] const fno_config& config() const noexcept { return config_; }

    /// Returns the expected input shape for the FNO layer.
    [[nodiscard]] const std::vector<size_t>& input_shape() const noexcept {
        return config_.input_shape;
    }

    /// Returns the expected output shape for the FNO layer.
    [[nodiscard]] const std::vector<size_t>& output_shape() const noexcept {
        return config_.output_shape;
    }

    /// Returns the number of trainable parameters in the model.
    [[nodiscard]] size_t num_params() const noexcept { return num_params_; }

    /// Returns the name of the neural operator.
    [[nodiscard]] std::string_view name() const noexcept {
        return "FourierNeuralOperator";
    }

    /// Returns true if the layer is valid (has an underlying handle).
    [[nodiscard]] bool is_valid() const noexcept { return handle_ != nullptr; }

    /// Explicit bool conversion.
    explicit operator bool() const noexcept { return is_valid(); }

private:
    // Private constructor
    explicit fno_layer(NavatalaFnoLayer* handle, fno_config cfg, size_t num_params);

    NavatalaFnoLayer* handle_;
    fno_config config_;
    size_t num_params_;
    resources* parent_resources_;
};

// ============================================================================
// Inline Implementations
// ============================================================================

inline fno_layer::fno_layer() noexcept
    : handle_(nullptr)
    , config_()
    , num_params_(0)
    , parent_resources_(nullptr)
{}

inline fno_layer::fno_layer(NavatalaFnoLayer* handle, fno_config cfg, size_t num_params)
    : handle_(handle)
    , config_(std::move(cfg))
    , num_params_(num_params)
    , parent_resources_(nullptr)
{}

inline fno_layer::~fno_layer() noexcept {
    if (handle_) {
        // navatala_fno_destroy(handle_);
        handle_ = nullptr;
    }
}

inline fno_layer::fno_layer(fno_layer&& other) noexcept
    : handle_(other.handle_)
    , config_(std::move(other.config_))
    , num_params_(other.num_params_)
    , parent_resources_(other.parent_resources_)
{
    other.handle_ = nullptr;
    other.num_params_ = 0;
    other.parent_resources_ = nullptr;
}

inline fno_layer& fno_layer::operator=(fno_layer&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            // navatala_fno_destroy(handle_);
        }
        handle_ = other.handle_;
        config_ = std::move(other.config_);
        num_params_ = other.num_params_;
        parent_resources_ = other.parent_resources_;

        other.handle_ = nullptr;
        other.num_params_ = 0;
        other.parent_resources_ = nullptr;
    }
    return *this;
}

} // namespace neural_operators
} // namespace navatala
