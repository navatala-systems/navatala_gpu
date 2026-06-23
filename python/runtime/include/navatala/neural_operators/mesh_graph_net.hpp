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

// Navatala C++ Wrapper Layer - Mesh Graph Network
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

/// @file navatala/neural_operators/mesh_graph_net.hpp
/// @brief GPU-accelerated Mesh Graph Network (MGN).
///
/// This header provides:
/// - mgn_config: Configuration for MeshGraphNet
/// - mesh_graph_net: RAII wrapper for MGN inference
///
/// MeshGraphNet is a message-passing neural network designed for mesh-based
/// physics simulations. It operates on graph representations of meshes where:
/// - Nodes represent mesh vertices with physical quantities
/// - Edges represent connectivity between vertices
///
/// @warning This module is EXPERIMENTAL. The API may change without notice.
///
/// Example:
/// @code
///   auto res = resources::create().value();
///
///   // Configure MGN
///   mgn_config cfg;
///   cfg.node_features = {6};    // Position (3) + velocity (3)
///   cfg.edge_features = {4};    // Relative position (3) + distance (1)
///   cfg.hidden_size = 128;
///   cfg.n_layers = 15;          // 15 message passing steps
///   cfg.output_size = 3;        // Acceleration prediction
///
///   // Create MGN
///   auto mgn = mesh_graph_net::create(res, cfg).value();
///
///   // Forward pass
///   auto output = mgn.forward(node_features, edge_index, edge_features, q);
/// @endcode

// Forward declare FFI types
struct NavatalaMeshGraphNet;

namespace navatala {
namespace neural_operators {

// ============================================================================
// MGN Configuration
// ============================================================================

/// @brief Configuration structure for the Mesh Graph Network (MGN).
struct mgn_config {
    std::vector<size_t> node_features;  ///< Shape of node feature vectors
    std::vector<size_t> edge_features;  ///< Shape of edge feature vectors

    size_t hidden_size = 128;   ///< Hidden layer dimension
    size_t output_size = 3;     ///< Output dimension per node (e.g., 3 for 3D acceleration)
    size_t n_layers = 15;       ///< Number of message passing layers

    bool use_edge_features = true;  ///< Whether to use edge features in message passing
    bool residual_connections = true; ///< Add residual connections
    bool layer_norm = true;     ///< Use layer normalization

    activation act = activation::gelu; ///< Activation function
    weight_init init = weight_init::xavier_uniform; ///< Weight initialization
    data_type dtype = data_type::float32; ///< Data type for computation

    // Message aggregation
    enum class aggregation : uint32_t {
        sum = 0,    ///< Sum aggregation
        mean = 1,   ///< Mean aggregation
        max = 2     ///< Max aggregation
    };
    aggregation agg = aggregation::sum; ///< Message aggregation method

    /// Returns expected node feature dimension.
    [[nodiscard]] size_t node_dim() const noexcept {
        return node_features.empty() ? 0 : node_features.back();
    }

    /// Returns expected edge feature dimension.
    [[nodiscard]] size_t edge_dim() const noexcept {
        return edge_features.empty() ? 0 : edge_features.back();
    }
};

// ============================================================================
// Mesh Graph Network Class
// ============================================================================

/// @brief RAII wrapper for a device-resident Mesh Graph Network.
///
/// MeshGraphNet performs iterative message passing on graphs:
/// 1. Edge Update: Compute messages using source/dest node features + edge features
/// 2. Aggregation: Aggregate incoming messages for each node (sum/mean/max)
/// 3. Node Update: Update node features using aggregated messages
/// 4. Repeat for n_layers
///
/// The architecture follows the encode-process-decode pattern:
/// - Encoder: Projects input features to hidden dimension
/// - Processor: Multiple message passing layers with residual connections
/// - Decoder: Projects hidden features to output dimension
///
/// Thread Safety: `forward()` is safe for concurrent calls after creation.
/// `create()`, `save()`, and `load()` are not thread-safe.
class mesh_graph_net {
public:
    // -------------------- Factory Methods --------------------

    /// Creates an MGN from a configuration.
    /// @param res GPU resources.
    /// @param cfg MGN configuration.
    /// @return An expected<mesh_graph_net> or an error.
    [[nodiscard]] static result<mesh_graph_net> create(
        resources& res,
        const mgn_config& cfg);

    /// Loads an MGN model from a file.
    /// @param res GPU resources.
    /// @param path Path to the model file.
    /// @param cfg MGN configuration (must match saved model).
    /// @return An expected<mesh_graph_net> or an error.
    [[nodiscard]] static result<mesh_graph_net> load(
        resources& res,
        const std::string& path,
        const mgn_config& cfg);

    // -------------------- Constructors/Destructor --------------------

    /// Default constructor for an invalid MGN.
    mesh_graph_net() noexcept;

    /// Destructor. Releases the underlying GPU MGN model.
    ~mesh_graph_net() noexcept;

    // Non-copyable
    mesh_graph_net(const mesh_graph_net&) = delete;
    mesh_graph_net& operator=(const mesh_graph_net&) = delete;

    // Movable
    mesh_graph_net(mesh_graph_net&& other) noexcept;
    mesh_graph_net& operator=(mesh_graph_net&& other) noexcept;

    // -------------------- Inference --------------------

    /// Performs a forward pass (inference) through the MGN.
    /// @param node_features Node feature tensor [N_nodes, node_dim].
    /// @param edge_index Edge connectivity [2, N_edges] (src_idx, dst_idx).
    /// @param edge_features Edge feature tensor [N_edges, edge_dim].
    /// @param q The queue to use for the operation.
    /// @return An expected<buffer> containing output [N_nodes, output_size] or an error.
    [[nodiscard]] result<buffer> forward(
        const buffer& node_features,
        const buffer& edge_index,
        const buffer& edge_features,
        const queue& q) const;

    /// Simplified forward pass without explicit edge features.
    /// Edge features are computed from node positions if available.
    /// @param node_features Node feature tensor [N_nodes, node_dim].
    /// @param edge_index Edge connectivity [2, N_edges].
    /// @param q The queue to use for the operation.
    /// @return An expected<buffer> containing output [N_nodes, output_size] or an error.
    [[nodiscard]] result<buffer> forward(
        const buffer& node_features,
        const buffer& edge_index,
        const queue& q) const;

    // -------------------- Graph Utilities --------------------

    /// Constructs edge index from mesh connectivity.
    /// @param faces Face connectivity [N_faces, vertices_per_face].
    /// @param res GPU resources.
    /// @return An expected<buffer> containing edge_index [2, N_edges].
    [[nodiscard]] static result<buffer> build_edge_index(
        const buffer& faces,
        resources& res);

    /// Computes edge features from node positions.
    /// Edge features include: relative position vector + distance.
    /// @param node_positions Node positions [N_nodes, 3].
    /// @param edge_index Edge connectivity [2, N_edges].
    /// @param res GPU resources.
    /// @return An expected<buffer> containing edge_features [N_edges, 4].
    [[nodiscard]] static result<buffer> compute_edge_features(
        const buffer& node_positions,
        const buffer& edge_index,
        resources& res);

    // -------------------- Persistence --------------------

    /// Saves the MGN model weights to a file.
    void save(const std::string& path) const;

    // -------------------- Accessors --------------------

    /// Returns the configuration used to create this MGN.
    [[nodiscard]] const mgn_config& config() const noexcept { return config_; }

    /// Returns the expected node feature shape.
    [[nodiscard]] const std::vector<size_t>& input_shape() const noexcept {
        return config_.node_features;
    }

    /// Returns the output shape per node.
    [[nodiscard]] const std::vector<size_t>& output_shape() const noexcept {
        return output_shape_;
    }

    /// Returns the number of trainable parameters.
    [[nodiscard]] size_t num_params() const noexcept { return num_params_; }

    /// Returns the name of the neural operator.
    [[nodiscard]] std::string_view name() const noexcept {
        return "MeshGraphNetwork";
    }

    /// Returns true if the model is valid.
    [[nodiscard]] bool is_valid() const noexcept { return handle_ != nullptr; }

    /// Explicit bool conversion.
    explicit operator bool() const noexcept { return is_valid(); }

private:
    // Private constructor
    explicit mesh_graph_net(NavatalaMeshGraphNet* handle, mgn_config cfg, size_t num_params);

    NavatalaMeshGraphNet* handle_;
    mgn_config config_;
    std::vector<size_t> output_shape_;
    size_t num_params_;
    resources* parent_resources_;
};

// ============================================================================
// Inline Implementations
// ============================================================================

inline mesh_graph_net::mesh_graph_net() noexcept
    : handle_(nullptr)
    , config_()
    , output_shape_()
    , num_params_(0)
    , parent_resources_(nullptr)
{}

inline mesh_graph_net::mesh_graph_net(NavatalaMeshGraphNet* handle, mgn_config cfg, size_t num_params)
    : handle_(handle)
    , config_(std::move(cfg))
    , output_shape_({config_.output_size})
    , num_params_(num_params)
    , parent_resources_(nullptr)
{}

inline mesh_graph_net::~mesh_graph_net() noexcept {
    if (handle_) {
        // navatala_mgn_destroy(handle_);
        handle_ = nullptr;
    }
}

inline mesh_graph_net::mesh_graph_net(mesh_graph_net&& other) noexcept
    : handle_(other.handle_)
    , config_(std::move(other.config_))
    , output_shape_(std::move(other.output_shape_))
    , num_params_(other.num_params_)
    , parent_resources_(other.parent_resources_)
{
    other.handle_ = nullptr;
    other.num_params_ = 0;
    other.parent_resources_ = nullptr;
}

inline mesh_graph_net& mesh_graph_net::operator=(mesh_graph_net&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            // navatala_mgn_destroy(handle_);
        }
        handle_ = other.handle_;
        config_ = std::move(other.config_);
        output_shape_ = std::move(other.output_shape_);
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
