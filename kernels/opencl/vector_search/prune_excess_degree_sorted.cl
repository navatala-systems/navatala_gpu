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

__kernel void navatala_vector_search_prune_excess_degree_sorted(__global uint* graph, __global const float* neighbor_distances, __global uint* degrees, __global const uint* n_vertices, __global const uint* current_max_degree, __global const uint* target_max_degree, __global const uint* invalid_id) {
  int gid0 = (int)get_global_id(0);
  uint vid = ((uint)((int)(get_group_id(0))));
  uint tid = ((uint)((int)(get_local_id(0))));
  uint nv = n_vertices[0];
  uint cmd = current_max_degree[0];
  uint tmd = target_max_degree[0];
  uint inv = invalid_id[0];
  __local uint shared_ids[128];
  __local float shared_dists[128];
  if ((vid < nv)) {
    uint deg = degrees[vid];
    if ((deg > tmd)) {
      if ((tid < deg)) {
        uint read_idx = ((vid * cmd) + tid);
        uint neighbor_id = graph[read_idx];
        float neighbor_dist = neighbor_distances[read_idx];
        shared_ids[tid] = neighbor_id;
        shared_dists[tid] = neighbor_dist;
      } else {
        if ((tid < (uint)(128u))) {
          shared_ids[tid] = inv;
          shared_dists[tid] = as_float(0x7e967699u);
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int stage = 0; stage < (int)((uint)(7u)); ++stage) {
        for (int substage = 0; substage < (int)((stage + (uint)(1u))); ++substage) {
          uint half_net = ((uint)(1u) << (stage - substage));
          uint partner = (tid ^ half_net);
          uint direction_bit = ((tid >> (stage + (uint)(1u))) & (uint)(1u));
          if ((partner < (uint)(128u))) {
            float my_dist = shared_dists[tid];
            float partner_d = shared_dists[partner];
            uint should_swap = (((tid < partner)) ? ((((direction_bit == (uint)(0u))) ? ((((my_dist > partner_d)) ? ((uint)(1u)) : ((uint)(0u)))) : ((((my_dist < partner_d)) ? ((uint)(1u)) : ((uint)(0u)))))) : ((uint)(0u)));
            if ((should_swap == (uint)(1u))) {
              uint my_id = shared_ids[tid];
              uint partner_id = shared_ids[partner];
              shared_ids[tid] = partner_id;
              shared_ids[partner] = my_id;
              shared_dists[tid] = partner_d;
              shared_dists[partner] = my_dist;
            }
          }
          barrier(CLK_LOCAL_MEM_FENCE);
        }
      }
      if ((tid < cmd)) {
        uint write_idx = ((vid * cmd) + tid);
        if ((tid < tmd)) {
          uint sorted_id = shared_ids[tid];
          graph[write_idx] = sorted_id;
        } else {
          graph[write_idx] = inv;
        }
      }
      if ((tid == (uint)(0u))) {
        degrees[vid] = tmd;
      }
    }
  }
}
