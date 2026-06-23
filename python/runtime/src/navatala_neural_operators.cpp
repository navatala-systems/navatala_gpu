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

// Navatala C++ Wrapper Layer - Neural Operators Implementation
// Part of the Navatala GPU Runtime Library
//
// WARNING: This module is EXPERIMENTAL. The API may change without notice.

#include "../include/navatala/neural_operators/fno.hpp"
#include "../include/navatala/neural_operators/pino.hpp"
#include "../include/navatala/neural_operators/mesh_graph_net.hpp"
#include "../include/navatala/buffer.hpp"
#include "../include/navatala/navatala_ffi.h"

#include <cstring>

namespace navatala {
namespace neural_operators {

// ============================================================================
// FNO Layer Implementation
// ============================================================================

result<fno_layer> fno_layer::create(resources& res, const fno_config& cfg) {
    // Build FFI config
    NavatalaFnoConfig ffi_cfg{};
    ffi_cfg.input_shape = cfg.input_shape.data();
    ffi_cfg.input_shape_len = cfg.input_shape.size();
    ffi_cfg.output_shape = cfg.output_shape.data();
    ffi_cfg.output_shape_len = cfg.output_shape.size();
    ffi_cfg.n_modes = cfg.n_modes;
    ffi_cfg.hidden_size = cfg.hidden_size;
    ffi_cfg.n_layers = cfg.n_layers;
    ffi_cfg.activation = static_cast<int>(cfg.act);
    ffi_cfg.padding = static_cast<int>(cfg.pad);
    ffi_cfg.factorized = cfg.factorized ? 1 : 0;
    ffi_cfg.skip_connections = cfg.skip_connections ? 1 : 0;
    ffi_cfg.weight_init = static_cast<int>(cfg.init);
    ffi_cfg.dtype = static_cast<int>(cfg.dtype);

    NavatalaFnoLayer* handle = nullptr;
    NavatalaErrorCode err = navatala_fno_create(res.handle(), &ffi_cfg, &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<fno_layer>(translate_error(err), "failed to create FNO layer");
    }

    size_t num_params = 0;
    navatala_fno_num_params(handle, &num_params);

    fno_layer layer(handle, cfg, num_params);
    layer.parent_resources_ = &res;
    return std::move(layer);
}

result<fno_layer> fno_layer::load(
    resources& res,
    const std::string& path,
    const fno_config& cfg)
{
    NavatalaFnoConfig ffi_cfg{};
    ffi_cfg.input_shape = cfg.input_shape.data();
    ffi_cfg.input_shape_len = cfg.input_shape.size();
    ffi_cfg.output_shape = cfg.output_shape.data();
    ffi_cfg.output_shape_len = cfg.output_shape.size();
    ffi_cfg.n_modes = cfg.n_modes;
    ffi_cfg.hidden_size = cfg.hidden_size;
    ffi_cfg.n_layers = cfg.n_layers;
    ffi_cfg.activation = static_cast<int>(cfg.act);
    ffi_cfg.padding = static_cast<int>(cfg.pad);
    ffi_cfg.factorized = cfg.factorized ? 1 : 0;
    ffi_cfg.skip_connections = cfg.skip_connections ? 1 : 0;
    ffi_cfg.weight_init = static_cast<int>(cfg.init);
    ffi_cfg.dtype = static_cast<int>(cfg.dtype);

    NavatalaFnoLayer* handle = nullptr;
    NavatalaErrorCode err = navatala_fno_load(res.handle(), path.c_str(), &ffi_cfg, &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<fno_layer>(translate_error(err), "failed to load FNO layer");
    }

    size_t num_params = 0;
    navatala_fno_num_params(handle, &num_params);

    fno_layer layer(handle, cfg, num_params);
    layer.parent_resources_ = &res;
    return std::move(layer);
}

result<buffer> fno_layer::forward(const buffer& input, const queue& q) const {
    if (!handle_) {
        return result<buffer>(error_code::invalid_handle, "FNO layer is not valid");
    }

    if (!parent_resources_) {
        return result<buffer>(error_code::invalid_handle, "FNO layer has no parent resources");
    }

    // Create output buffer with output shape
    auto output_result = buffer::create(*parent_resources_, config_.output_shape, config_.dtype);
    if (!output_result.has_value()) {
        return output_result;
    }

    auto output = std::move(output_result.value());

    // Get batch size (assume 1 for unbatched input)
    size_t batch_size = 1;

    NavatalaErrorCode err = navatala_fno_forward(
        handle_,
        input.data(),
        batch_size,
        output.data(),
        q.handle()
    );

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "FNO forward pass failed");
    }

    return std::move(output);
}

result<buffer> fno_layer::forward(
    const buffer& input,
    const buffer& params,
    const queue& q) const
{
    if (!handle_) {
        return result<buffer>(error_code::invalid_handle, "FNO layer is not valid");
    }

    if (!parent_resources_) {
        return result<buffer>(error_code::invalid_handle, "FNO layer has no parent resources");
    }

    auto output_result = buffer::create(*parent_resources_, config_.output_shape, config_.dtype);
    if (!output_result.has_value()) {
        return output_result;
    }

    auto output = std::move(output_result.value());
    size_t batch_size = 1;

    NavatalaErrorCode err = navatala_fno_forward_with_params(
        handle_,
        input.data(),
        params.data(),
        batch_size,
        output.data(),
        q.handle()
    );

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "FNO forward pass with params failed");
    }

    return std::move(output);
}

void fno_layer::save(const std::string& path) const {
    if (handle_) {
        navatala_fno_save(handle_, path.c_str());
    }
}

// ============================================================================
// PINO Layer Implementation
// ============================================================================

result<pino_layer> pino_layer::create(resources& res, const pino_config& cfg) {
    NavatalaPINOConfig ffi_cfg{};
    ffi_cfg.input_shape = cfg.input_shape.data();
    ffi_cfg.input_shape_len = cfg.input_shape.size();
    ffi_cfg.output_shape = cfg.output_shape.data();
    ffi_cfg.output_shape_len = cfg.output_shape.size();
    ffi_cfg.pde_type = static_cast<int>(cfg.pde);
    ffi_cfg.n_collocation_points = cfg.n_collocation_points;
    ffi_cfg.physics_loss_weight = cfg.physics_loss_weight;
    ffi_cfg.data_loss_weight = cfg.data_loss_weight;
    ffi_cfg.viscosity = cfg.viscosity;
    ffi_cfg.diffusivity = cfg.diffusivity;
    ffi_cfg.wave_speed = cfg.wave_speed;
    ffi_cfg.periodic_bc = cfg.periodic_bc ? 1 : 0;
    ffi_cfg.bc_loss_weight = cfg.bc_loss_weight;
    ffi_cfg.n_modes = cfg.n_modes;
    ffi_cfg.hidden_size = cfg.hidden_size;
    ffi_cfg.n_layers = cfg.n_layers;
    ffi_cfg.activation = static_cast<int>(cfg.act);

    NavatalaPINOLayer* handle = nullptr;
    NavatalaErrorCode err = navatala_pino_create(res.handle(), &ffi_cfg, &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<pino_layer>(translate_error(err), "failed to create PINO layer");
    }

    pino_layer layer(handle, cfg);
    layer.parent_resources_ = &res;
    return std::move(layer);
}

result<pino_layer> pino_layer::load(
    resources& res,
    const std::string& path,
    const pino_config& cfg)
{
    NavatalaPINOConfig ffi_cfg{};
    ffi_cfg.input_shape = cfg.input_shape.data();
    ffi_cfg.input_shape_len = cfg.input_shape.size();
    ffi_cfg.output_shape = cfg.output_shape.data();
    ffi_cfg.output_shape_len = cfg.output_shape.size();
    ffi_cfg.pde_type = static_cast<int>(cfg.pde);
    ffi_cfg.n_collocation_points = cfg.n_collocation_points;
    ffi_cfg.physics_loss_weight = cfg.physics_loss_weight;
    ffi_cfg.data_loss_weight = cfg.data_loss_weight;
    ffi_cfg.viscosity = cfg.viscosity;
    ffi_cfg.diffusivity = cfg.diffusivity;
    ffi_cfg.wave_speed = cfg.wave_speed;
    ffi_cfg.periodic_bc = cfg.periodic_bc ? 1 : 0;
    ffi_cfg.bc_loss_weight = cfg.bc_loss_weight;
    ffi_cfg.n_modes = cfg.n_modes;
    ffi_cfg.hidden_size = cfg.hidden_size;
    ffi_cfg.n_layers = cfg.n_layers;
    ffi_cfg.activation = static_cast<int>(cfg.act);

    NavatalaPINOLayer* handle = nullptr;
    NavatalaErrorCode err = navatala_pino_load(res.handle(), path.c_str(), &ffi_cfg, &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<pino_layer>(translate_error(err), "failed to load PINO layer");
    }

    pino_layer layer(handle, cfg);
    layer.parent_resources_ = &res;
    return std::move(layer);
}

result<buffer> pino_layer::forward(const buffer& input, const queue& q) const {
    if (!handle_) {
        return result<buffer>(error_code::invalid_handle, "PINO layer is not valid");
    }

    if (!parent_resources_) {
        return result<buffer>(error_code::invalid_handle, "PINO layer has no parent resources");
    }

    auto output_result = buffer::create(*parent_resources_, config_.output_shape, data_type::float32);
    if (!output_result.has_value()) {
        return output_result;
    }

    auto output = std::move(output_result.value());
    size_t batch_size = 1;

    NavatalaErrorCode err = navatala_pino_forward(
        handle_,
        input.data(),
        batch_size,
        output.data(),
        q.handle()
    );

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "PINO forward pass failed");
    }

    return std::move(output);
}

result<pino_train_result> pino_layer::train_step(
    const buffer& input,
    const buffer& ground_truth,
    const queue& q)
{
    if (!handle_) {
        return result<pino_train_result>(error_code::invalid_handle, "PINO layer is not valid");
    }

    pino_train_result res_out{};
    size_t batch_size = 1;

    NavatalaErrorCode err = navatala_pino_train_step(
        handle_,
        input.data(),
        ground_truth.data(),
        batch_size,
        &res_out.total_loss,
        &res_out.data_loss,
        &res_out.physics_loss,
        &res_out.bc_loss,
        q.handle()
    );

    if (err != NAVATALA_SUCCESS) {
        return result<pino_train_result>(translate_error(err), "PINO train step failed");
    }

    training_steps_++;
    res_out.step = training_steps_;

    return res_out;
}

result<float> pino_layer::compute_pde_residual(
    const buffer& solution,
    const queue& q) const
{
    if (!handle_) {
        return result<float>(error_code::invalid_handle, "PINO layer is not valid");
    }

    float mean_residual = 0.0f;
    size_t batch_size = 1;

    NavatalaErrorCode err = navatala_pino_compute_residual(
        handle_,
        solution.data(),
        batch_size,
        &mean_residual,
        q.handle()
    );

    if (err != NAVATALA_SUCCESS) {
        return result<float>(translate_error(err), "PINO compute residual failed");
    }

    return mean_residual;
}

void pino_layer::save(const std::string& path) const {
    if (handle_) {
        navatala_pino_save(handle_, path.c_str());
    }
}

// ============================================================================
// Mesh Graph Network Implementation
// ============================================================================

result<mesh_graph_net> mesh_graph_net::create(resources& res, const mgn_config& cfg) {
    NavatalaMGNConfig ffi_cfg{};
    ffi_cfg.node_features = cfg.node_features.data();
    ffi_cfg.node_features_len = cfg.node_features.size();
    ffi_cfg.edge_features = cfg.edge_features.data();
    ffi_cfg.edge_features_len = cfg.edge_features.size();
    ffi_cfg.hidden_size = cfg.hidden_size;
    ffi_cfg.output_size = cfg.output_size;
    ffi_cfg.n_layers = cfg.n_layers;
    ffi_cfg.use_edge_features = cfg.use_edge_features ? 1 : 0;
    ffi_cfg.residual_connections = cfg.residual_connections ? 1 : 0;
    ffi_cfg.layer_norm = cfg.layer_norm ? 1 : 0;
    ffi_cfg.activation = static_cast<int>(cfg.act);
    ffi_cfg.aggregation = static_cast<int>(cfg.agg);
    ffi_cfg.dtype = static_cast<int>(cfg.dtype);

    NavatalaMeshGraphNet* handle = nullptr;
    NavatalaErrorCode err = navatala_mgn_create(res.handle(), &ffi_cfg, &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<mesh_graph_net>(translate_error(err), "failed to create MGN");
    }

    size_t num_params = 0;
    navatala_mgn_num_params(handle, &num_params);

    mesh_graph_net mgn(handle, cfg, num_params);
    mgn.parent_resources_ = &res;
    return std::move(mgn);
}

result<mesh_graph_net> mesh_graph_net::load(
    resources& res,
    const std::string& path,
    const mgn_config& cfg)
{
    NavatalaMGNConfig ffi_cfg{};
    ffi_cfg.node_features = cfg.node_features.data();
    ffi_cfg.node_features_len = cfg.node_features.size();
    ffi_cfg.edge_features = cfg.edge_features.data();
    ffi_cfg.edge_features_len = cfg.edge_features.size();
    ffi_cfg.hidden_size = cfg.hidden_size;
    ffi_cfg.output_size = cfg.output_size;
    ffi_cfg.n_layers = cfg.n_layers;
    ffi_cfg.use_edge_features = cfg.use_edge_features ? 1 : 0;
    ffi_cfg.residual_connections = cfg.residual_connections ? 1 : 0;
    ffi_cfg.layer_norm = cfg.layer_norm ? 1 : 0;
    ffi_cfg.activation = static_cast<int>(cfg.act);
    ffi_cfg.aggregation = static_cast<int>(cfg.agg);
    ffi_cfg.dtype = static_cast<int>(cfg.dtype);

    NavatalaMeshGraphNet* handle = nullptr;
    NavatalaErrorCode err = navatala_mgn_load(res.handle(), path.c_str(), &ffi_cfg, &handle);

    if (err != NAVATALA_SUCCESS) {
        return result<mesh_graph_net>(translate_error(err), "failed to load MGN");
    }

    size_t num_params = 0;
    navatala_mgn_num_params(handle, &num_params);

    mesh_graph_net mgn(handle, cfg, num_params);
    mgn.parent_resources_ = &res;
    return std::move(mgn);
}

result<buffer> mesh_graph_net::forward(
    const buffer& node_features,
    const buffer& edge_index,
    const buffer& edge_features,
    const queue& q) const
{
    if (!handle_) {
        return result<buffer>(error_code::invalid_handle, "MGN is not valid");
    }

    if (!parent_resources_) {
        return result<buffer>(error_code::invalid_handle, "MGN has no parent resources");
    }

    // Infer n_nodes from buffer size and node dimension
    size_t node_dim = config_.node_dim();
    size_t n_nodes = node_dim > 0 ? node_features.bytes() / (node_dim * sizeof(float)) : 0;
    size_t n_edges = edge_index.bytes() / (2 * sizeof(uint32_t));

    // Create output buffer [n_nodes, output_size]
    std::vector<size_t> output_shape = {n_nodes, config_.output_size};
    auto output_result = buffer::create(*parent_resources_, output_shape, config_.dtype);
    if (!output_result.has_value()) {
        return output_result;
    }

    auto output = std::move(output_result.value());

    NavatalaErrorCode err = navatala_mgn_forward(
        handle_,
        node_features.data(),
        n_nodes,
        edge_index.data(),
        n_edges,
        edge_features.data(),
        output.data(),
        q.handle()
    );

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "MGN forward pass failed");
    }

    return std::move(output);
}

result<buffer> mesh_graph_net::forward(
    const buffer& node_features,
    const buffer& edge_index,
    const queue& q) const
{
    if (!handle_) {
        return result<buffer>(error_code::invalid_handle, "MGN is not valid");
    }

    if (!parent_resources_) {
        return result<buffer>(error_code::invalid_handle, "MGN has no parent resources");
    }

    size_t node_dim = config_.node_dim();
    size_t n_nodes = node_dim > 0 ? node_features.bytes() / (node_dim * sizeof(float)) : 0;
    size_t n_edges = edge_index.bytes() / (2 * sizeof(uint32_t));

    std::vector<size_t> output_shape = {n_nodes, config_.output_size};
    auto output_result = buffer::create(*parent_resources_, output_shape, config_.dtype);
    if (!output_result.has_value()) {
        return output_result;
    }

    auto output = std::move(output_result.value());

    NavatalaErrorCode err = navatala_mgn_forward_no_edge_features(
        handle_,
        node_features.data(),
        n_nodes,
        edge_index.data(),
        n_edges,
        output.data(),
        q.handle()
    );

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "MGN forward pass (no edge features) failed");
    }

    return std::move(output);
}

result<buffer> mesh_graph_net::build_edge_index(
    const buffer& faces,
    resources& res)
{
    // Assume triangular faces for now
    size_t vertices_per_face = 3;
    size_t n_faces = faces.bytes() / (vertices_per_face * sizeof(uint32_t));

    // Maximum edges: 2 * vertices_per_face * n_faces (bidirectional)
    size_t max_edges = 2 * vertices_per_face * n_faces;
    std::vector<size_t> edge_shape = {2, max_edges};  // [2, n_edges] format

    auto edge_index_result = buffer::create(res, edge_shape, data_type::uint32);
    if (!edge_index_result.has_value()) {
        return edge_index_result;
    }

    auto edge_index = std::move(edge_index_result.value());

    size_t n_edges = 0;
    NavatalaErrorCode err = navatala_mgn_build_edge_index(
        res.handle(),
        faces.data(),
        n_faces,
        vertices_per_face,
        edge_index.data(),
        &n_edges,
        nullptr
    );

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "MGN build edge index failed");
    }

    return std::move(edge_index);
}

result<buffer> mesh_graph_net::compute_edge_features(
    const buffer& node_positions,
    const buffer& edge_index,
    resources& res)
{
    size_t n_nodes = node_positions.bytes() / (3 * sizeof(float));
    size_t n_edges = edge_index.bytes() / (2 * sizeof(uint32_t));

    // Edge features: [n_edges, 4] (dx, dy, dz, distance)
    std::vector<size_t> edge_feature_shape = {n_edges, 4};
    auto edge_features_result = buffer::create(res, edge_feature_shape, data_type::float32);
    if (!edge_features_result.has_value()) {
        return edge_features_result;
    }

    auto edge_features = std::move(edge_features_result.value());

    NavatalaErrorCode err = navatala_mgn_compute_edge_features(
        res.handle(),
        node_positions.data(),
        n_nodes,
        edge_index.data(),
        n_edges,
        edge_features.data(),
        nullptr
    );

    if (err != NAVATALA_SUCCESS) {
        return result<buffer>(translate_error(err), "MGN compute edge features failed");
    }

    return std::move(edge_features);
}

void mesh_graph_net::save(const std::string& path) const {
    if (handle_) {
        navatala_mgn_save(handle_, path.c_str());
    }
}

} // namespace neural_operators
} // namespace navatala
