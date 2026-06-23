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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_vector_search_prune_graph_r_n_g_f32(const float* dataset, const unsigned int* graph, const float* graph_distances, const unsigned int* n_nodes, const unsigned int* graph_degree, const unsigned int* dim, unsigned int* pruned_graph, float* pruned_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int deg = graph_degree[0];
  unsigned int d = dim[0];
  unsigned int total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int node_idx = (gid / deg);
    unsigned int slot = (gid % deg);
    unsigned int graph_base = (node_idx * deg);
    unsigned int target = graph[gid];
    float edge_dist = graph_distances[gid];
    unsigned int target_base = (target * d);
    bool should_prune = false;
    for (int wi = 0; wi < (int)(deg); ++wi) {
      bool is_self_slot = (wi == slot);
      if (wi != slot) {
        unsigned int w_slot = (graph_base + wi);
        unsigned int w = graph[w_slot];
        float d_uw = graph_distances[w_slot];
        bool uw_closer = (d_uw < edge_dist);
        if (uw_closer) {
          unsigned int w_base = (w * d);
          float d_wv_acc = __uint_as_float(0x00000000u);
          for (int k = 0; k < (int)(d); ++k) {
            unsigned int w_off = (w_base + k);
            unsigned int t_off = (target_base + k);
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
    unsigned int invalid_marker = 4294967295u;
    unsigned int out_idx = ((prune) ? (invalid_marker) : (target));
    float out_dist = ((prune) ? (__uint_as_float(0x7f7fc99eu)) : (edge_dist));
    pruned_graph[gid] = out_idx;
    pruned_distances[gid] = out_dist;
  }
}
