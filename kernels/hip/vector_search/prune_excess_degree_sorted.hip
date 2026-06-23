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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_prune_excess_degree_sorted(unsigned int* graph, const float* neighbor_distances, unsigned int* degrees, const unsigned int* n_vertices, const unsigned int* current_max_degree, const unsigned int* target_max_degree, const unsigned int* invalid_id) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x)));
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int nv = n_vertices[0];
  unsigned int cmd = current_max_degree[0];
  unsigned int tmd = target_max_degree[0];
  unsigned int inv = invalid_id[0];
  __shared__ unsigned int shared_ids[128];
  __shared__ float shared_dists[128];
  if (vid < nv) {
    unsigned int deg = degrees[vid];
    if (deg > tmd) {
      if (tid < deg) {
        unsigned int read_idx = ((vid * cmd) + tid);
        unsigned int neighbor_id = graph[read_idx];
        float neighbor_dist = neighbor_distances[read_idx];
        shared_ids[tid] = neighbor_id;
        shared_dists[tid] = neighbor_dist;
      } else {
        if (tid < 128u) {
          shared_ids[tid] = inv;
          shared_dists[tid] = __uint_as_float(0x7e967699u);
        }
      }
      __syncthreads();
      for (int stage = 0; stage < (int)(7u); ++stage) {
        for (int substage = 0; substage < (int)((stage + 1u)); ++substage) {
          unsigned int half_net = (1u << (stage - substage));
          unsigned int partner = (tid ^ half_net);
          unsigned int direction_bit = ((tid >> (stage + 1u)) & 1u);
          if (partner < 128u) {
            float my_dist = shared_dists[tid];
            float partner_d = shared_dists[partner];
            unsigned int should_swap = (((tid < partner)) ? ((((direction_bit == 0u)) ? ((((my_dist > partner_d)) ? (1u) : (0u))) : ((((my_dist < partner_d)) ? (1u) : (0u))))) : (0u));
            if (should_swap == 1u) {
              unsigned int my_id = shared_ids[tid];
              unsigned int partner_id = shared_ids[partner];
              shared_ids[tid] = partner_id;
              shared_ids[partner] = my_id;
              shared_dists[tid] = partner_d;
              shared_dists[partner] = my_dist;
            }
          }
          __syncthreads();
        }
      }
      if (tid < cmd) {
        unsigned int write_idx = ((vid * cmd) + tid);
        if (tid < tmd) {
          unsigned int sorted_id = shared_ids[tid];
          graph[write_idx] = sorted_id;
        } else {
          graph[write_idx] = inv;
        }
      }
      if (tid == 0u) {
        degrees[vid] = tmd;
      }
    }
  }
}
