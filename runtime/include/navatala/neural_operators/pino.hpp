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

// Navatala C++ Wrapper Layer - Physics-Informed Neural Operator
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

/// @file navatala/neural_operators/pino.hpp
/// @brief GPU-accelerated Physics-Informed Neural Operator (PINO).
///
/// This header provides:
/// - pino_config: Configuration for PINO layers
/// - pino_layer: RAII wrapper for PINO inference and training
///
/// PINO extends the Fourier Neural Operator with physics-informed training.
/// It uses automatic differentiation to compute PDE residuals in the spectral
/// domain, enabling data-efficient learning of physics-governed systems.
///
/// @warning This module is EXPERIMENTAL. The API may change without notice.
///
/// Example:
/// @code
///   auto res = resources::create().value();
///
///   // Configure PINO for Navier-Stokes
///   pino_config cfg;
///   cfg.input_shape = {64, 64, 2};    // Velocity field
///   cfg.output_shape = {64, 64, 2};
///   cfg.pde_type = pde_type::navier_stokes;
///   cfg.physics_loss_weight = 0.7f;
///   cfg.data_loss_weight = 0.3f;
///
///   // Create PINO layer
///   auto pino = pino_layer::create(res, cfg).value();
///
///   // Training step
///   auto loss = pino.train_step(input, ground_truth, res.default_queue());
///
///   // Inference
///   auto output = pino.forward(input, res.default_queue());
/// @endcode

// Forward declare FFI types
struct NavatalaPINOLayer;

namespace navatala {
namespace neural_operators {

// ============================================================================
// PDE Types
// ============================================================================

/// @brief Types of PDEs supported by PINO.
enum class pde_type : uint32_t {
    custom = 0,         ///< User-defined PDE (requires callback)
    navier_stokes = 1,  ///< Incompressible Navier-Stokes
    heat_equation = 2,  ///< Heat/diffusion equation
    wave_equation = 3,  ///< Wave equation
    burgers = 4,        ///< Burgers equation
    darcy_flow = 5,     ///< Darcy flow (porous media)
    poisson = 6,        ///< Poisson equation
    advection = 7       ///< Advection equation
};

/// @brief Returns the name of a PDE type.
[[nodiscard]] constexpr const char* pde_type_name(pde_type pde) noexcept {
    switch (pde) {
        case pde_type::custom:        return "custom";
        case pde_type::navier_stokes: return "navier_stokes";
        case pde_type::heat_equation: return "heat_equation";
        case pde_type::wave_equation: return "wave_equation";
        case pde_type::burgers:       return "burgers";
        case pde_type::darcy_flow:    return "darcy_flow";
        case pde_type::poisson:       return "poisson";
        case pde_type::advection:     return "advection";
        default:                      return "unknown";
    }
}

// ============================================================================
// PINO Configuration
// ============================================================================

/// @brief Configuration structure for the Physics-Informed Neural Operator (PINO).
struct pino_config {
    std::vector<size_t> input_shape;   ///< Input tensor shape
    std::vector<size_t> output_shape;  ///< Output tensor shape

    // PDE-specific parameters
    pde_type pde = pde_type::navier_stokes; ///< Type of PDE to solve
    size_t n_collocation_points = 1000; ///< Number of collocation points for physics loss
    float physics_loss_weight = 0.7f;   ///< Weight for the physics-based loss term
    float data_loss_weight = 0.3f;      ///< Weight for the data-based loss term

    // Physical parameters (PDE-specific)
    float viscosity = 0.01f;    ///< Kinematic viscosity (for Navier-Stokes)
    float diffusivity = 0.1f;   ///< Thermal diffusivity (for heat equation)
    float wave_speed = 1.0f;    ///< Wave propagation speed

    // Boundary condition parameters
    bool periodic_bc = true;    ///< Use periodic boundary conditions
    float bc_loss_weight = 0.1f; ///< Weight for boundary condition loss

    // Network architecture parameters (same as FNO)
    size_t n_modes = 12;        ///< Number of Fourier modes
    size_t hidden_size = 128;   ///< Hidden layer size
    size_t n_layers = 4;        ///< Number of spectral layers
    activation act = activation::tanh; ///< Activation function (tanh recommended for PDEs)

    // Training options
    training_config train;      ///< Training configuration
};

// ============================================================================
// PINO Training Result
// ============================================================================

/// @brief Result of a PINO training step.
struct pino_train_result {
    float total_loss = 0.0f;     ///< Combined loss
    float data_loss = 0.0f;      ///< Data fitting loss
    float physics_loss = 0.0f;   ///< PDE residual loss
    float bc_loss = 0.0f;        ///< Boundary condition loss
    size_t step = 0;             ///< Current training step
};

// ============================================================================
// PINO Layer Class
// ============================================================================

/// @brief RAII wrapper for a device-resident Physics-Informed Neural Operator layer.
///
/// PINO extends FNO with physics-informed training:
/// 1. Forward pass through FNO architecture
/// 2. Compute PDE residuals using spectral differentiation
/// 3. Combine data loss and physics loss for training
///
/// Spectral differentiation enables efficient gradient computation:
/// - First derivative: multiply by (2*pi*i*k) in Fourier space
/// - Second derivative: multiply by -(2*pi*k)^2
///
/// Thread Safety: `forward()` is safe for concurrent calls. `train_step()`,
/// `create()`, `save()`, and `load()` are not thread-safe.
class pino_layer {
public:
    // -------------------- Factory Methods --------------------

    /// Creates a PINO layer from a configuration.
    /// @param res GPU resources.
    /// @param cfg PINO configuration.
    /// @return An expected<pino_layer> or an error.
    [[nodiscard]] static result<pino_layer> create(
        resources& res,
        const pino_config& cfg);

    /// Loads a PINO model from a file.
    /// @param res GPU resources.
    /// @param path Path to the model file.
    /// @param cfg PINO configuration (must match saved model).
    /// @return An expected<pino_layer> or an error.
    [[nodiscard]] static result<pino_layer> load(
        resources& res,
        const std::string& path,
        const pino_config& cfg);

    // -------------------- Constructors/Destructor --------------------

    /// Default constructor for an invalid PINO layer.
    pino_layer() noexcept;

    /// Destructor. Releases the underlying GPU PINO layer.
    ~pino_layer() noexcept;

    // Non-copyable
    pino_layer(const pino_layer&) = delete;
    pino_layer& operator=(const pino_layer&) = delete;

    // Movable
    pino_layer(pino_layer&& other) noexcept;
    pino_layer& operator=(pino_layer&& other) noexcept;

    // -------------------- Inference --------------------

    /// Performs a forward pass (inference) through the PINO layer.
    /// @param input The input tensor on device.
    /// @param q The queue to use for the operation.
    /// @return An expected<buffer> containing the predicted solution or an error.
    [[nodiscard]] result<buffer> forward(const buffer& input, const queue& q) const;

    // -------------------- Training --------------------

    /// Performs a training step for the PINO model.
    /// Computes both data loss (against ground truth) and physics loss (PDE residual).
    /// @param input The input tensor for training.
    /// @param ground_truth The ground truth data for data loss.
    /// @param q The queue to use for the operation.
    /// @return An expected<pino_train_result> containing losses or an error.
    [[nodiscard]] result<pino_train_result> train_step(
        const buffer& input,
        const buffer& ground_truth,
        const queue& q);

    /// Computes PDE residual for a given solution (without updating weights).
    /// Useful for validation and monitoring training progress.
    /// @param solution The predicted or ground-truth solution.
    /// @param q The queue to use.
    /// @return An expected<float> containing the mean PDE residual or an error.
    [[nodiscard]] result<float> compute_pde_residual(
        const buffer& solution,
        const queue& q) const;

    // -------------------- Persistence --------------------

    /// Saves the PINO model weights to a file.
    void save(const std::string& path) const;

    // -------------------- Accessors --------------------

    /// Returns the configuration used to create this PINO layer.
    [[nodiscard]] const pino_config& config() const noexcept { return config_; }

    /// Returns the expected input shape.
    [[nodiscard]] const std::vector<size_t>& input_shape() const noexcept {
        return config_.input_shape;
    }

    /// Returns the expected output shape.
    [[nodiscard]] const std::vector<size_t>& output_shape() const noexcept {
        return config_.output_shape;
    }

    /// Returns the PDE type.
    [[nodiscard]] pde_type pde() const noexcept { return config_.pde; }

    /// Returns the name of the neural operator.
    [[nodiscard]] std::string_view name() const noexcept {
        return "PhysicsInformedNeuralOperator";
    }

    /// Returns the number of training steps completed.
    [[nodiscard]] size_t training_steps() const noexcept { return training_steps_; }

    /// Returns true if the layer is valid.
    [[nodiscard]] bool is_valid() const noexcept { return handle_ != nullptr; }

    /// Explicit bool conversion.
    explicit operator bool() const noexcept { return is_valid(); }

private:
    // Private constructor
    explicit pino_layer(NavatalaPINOLayer* handle, pino_config cfg);

    NavatalaPINOLayer* handle_;
    pino_config config_;
    size_t training_steps_;
    resources* parent_resources_;
};

// ============================================================================
// Inline Implementations
// ============================================================================

inline pino_layer::pino_layer() noexcept
    : handle_(nullptr)
    , config_()
    , training_steps_(0)
    , parent_resources_(nullptr)
{}

inline pino_layer::pino_layer(NavatalaPINOLayer* handle, pino_config cfg)
    : handle_(handle)
    , config_(std::move(cfg))
    , training_steps_(0)
    , parent_resources_(nullptr)
{}

inline pino_layer::~pino_layer() noexcept {
    if (handle_) {
        // navatala_pino_destroy(handle_);
        handle_ = nullptr;
    }
}

inline pino_layer::pino_layer(pino_layer&& other) noexcept
    : handle_(other.handle_)
    , config_(std::move(other.config_))
    , training_steps_(other.training_steps_)
    , parent_resources_(other.parent_resources_)
{
    other.handle_ = nullptr;
    other.training_steps_ = 0;
    other.parent_resources_ = nullptr;
}

inline pino_layer& pino_layer::operator=(pino_layer&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            // navatala_pino_destroy(handle_);
        }
        handle_ = other.handle_;
        config_ = std::move(other.config_);
        training_steps_ = other.training_steps_;
        parent_resources_ = other.parent_resources_;

        other.handle_ = nullptr;
        other.training_steps_ = 0;
        other.parent_resources_ = nullptr;
    }
    return *this;
}

} // namespace neural_operators
} // namespace navatala
