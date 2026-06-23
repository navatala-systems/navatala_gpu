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

__kernel void navatala_vector_search_prune_graph_r_n_g_f32(__global const float* dataset, __global const uint* graph, __global const float* graph_distances, __global const uint* n_nodes, __global const uint* graph_degree, __global const uint* dim, __global uint* pruned_graph, __global float* pruned_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  uint d = dim[0];
  uint total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint node_idx = (gid / deg);
    uint slot = (gid % deg);
    uint graph_base = (node_idx * deg);
    uint target = graph[gid];
    float edge_dist = graph_distances[gid];
    uint target_base = (target * d);
    bool should_prune = false;
    for (int wi = 0; wi < (int)(deg); ++wi) {
      bool is_self_slot = (wi == slot);
      if (wi != slot) {
        uint w_slot = (graph_base + wi);
        uint w = graph[w_slot];
        float d_uw = graph_distances[w_slot];
        bool uw_closer = (d_uw < edge_dist);
        if (uw_closer) {
          uint w_base = (w * d);
          float d_wv_acc = as_float(0x00000000u);
          for (int k = 0; k < (int)(d); ++k) {
            uint w_off = (w_base + k);
            uint t_off = (target_base + k);
            float w_val = dataset[w_off];
            float t_val = dataset[t_off];
            float diff = (w_val - t_val);
            float sq = (diff * diff);
            float old_wv = d_wv_acc;
            d_wv_acc = (old_wv + sq);
          }
          float d_wv = d_wv_acc;
          bool wv_closer = (d_wv < edge_dist);
          if (wv_closer) {
            should_prune = true;
          }
        }
      }
    }
    bool prune = should_prune;
    uint invalid_marker = (uint)(4294967295u);
    uint out_idx = ((prune) ? (invalid_marker) : (target));
    float out_dist = ((prune) ? (as_float(0x7f7fc99eu)) : (edge_dist));
    pruned_graph[gid] = out_idx;
    pruned_distances[gid] = out_dist;
  }
}
