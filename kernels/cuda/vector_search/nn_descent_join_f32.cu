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
extern "C" __global__ void navatala_vector_search_nn_descent_join_f32(const float* dataset, const unsigned int* graph, const float* graph_distances, const unsigned int* n_nodes, const unsigned int* graph_degree, const unsigned int* dim, unsigned int* new_graph, float* new_distances, unsigned int* update_count) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int node_idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int deg = graph_degree[0];
  unsigned int d = dim[0];
  bool inBounds = (node_idx < nn);
  if (inBounds) {
    unsigned int node_base = (node_idx * d);
    unsigned int graph_base = (node_idx * deg);
    float worst_dist = __uint_as_float(0x00000000u);
    unsigned int worst_slot = 0u;
    for (int i = 0; i < (int)(deg); ++i) {
      unsigned int slot_idx = (graph_base + i);
      float slot_dist = graph_distances[slot_idx];
      float cur_worst = worst_dist;
      bool is_worse = (slot_dist > cur_worst);
      if (is_worse) {
        worst_dist = slot_dist;
        worst_slot = i;
      }
    }
    for (int ni = 0; ni < (int)(deg); ++ni) {
      unsigned int neighbor_idx_slot = (graph_base + ni);
      unsigned int neighbor = graph[neighbor_idx_slot];
      unsigned int neighbor_graph_base = (neighbor * deg);
      for (int nni = 0; nni < (int)(deg); ++nni) {
        unsigned int nn_slot = (neighbor_graph_base + nni);
        unsigned int candidate = graph[nn_slot];
        bool is_self = (candidate == node_idx);
        if ((candidate != node_idx) && true) {
          unsigned int cand_base = (candidate * d);
          float cand_dist_acc = __uint_as_float(0x00000000u);
          for (int k = 0; k < (int)(d); ++k) {
            unsigned int n_off = (node_base + k);
            unsigned int c_off = (cand_base + k);
            float n_val = dataset[n_off];
            float c_val = dataset[c_off];
            float diff = (n_val - c_val);
            float sq = (diff * diff);
            float old_cd = cand_dist_acc;
            cand_dist_acc = (old_cd + sq);
          }
          float cand_dist = cand_dist_acc;
          float cur_worst2 = worst_dist;
          bool is_better = (cand_dist < cur_worst2);
          if (is_better) {
            unsigned int update_slot = worst_slot;
            unsigned int update_idx = (graph_base + update_slot);
            new_graph[update_idx] = candidate;
            new_distances[update_idx] = cand_dist;
            worst_dist = cand_dist;
            unsigned int _upd = atomicAdd(&(update_count[0u]), 1u);
          }
        }
      }
    }
  }
}
