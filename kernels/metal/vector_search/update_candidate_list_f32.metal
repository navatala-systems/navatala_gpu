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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_update_candidate_list_f32(device uint* candidates [[buffer(0)]], device float* distances [[buffer(1)]], device const uint* new_candidates [[buffer(2)]], device const float* new_distances [[buffer(3)]], device const uint* n_current [[buffer(4)]], device const uint* n_new [[buffer(5)]], device const uint* ef [[buffer(6)]], device uint* n_merged [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  uint n_curr = n_current[0];
  uint n_new_val = n_new[0];
  uint ef_val = ef[0];
  threadgroup uint merge_ids[512];
  threadgroup float merge_dists[512];
  if ((tid < n_curr)) {
    uint cid = candidates[tid];
    float cdist = distances[tid];
    merge_ids[tid] = cid;
    merge_dists[tid] = cdist;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((tid < n_new_val)) {
    uint offset = (n_curr + tid);
    uint nid = new_candidates[tid];
    float ndist = new_distances[tid];
    merge_ids[offset] = nid;
    merge_dists[offset] = ndist;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((tid == 0u)) {
    uint total = (n_curr + n_new_val);
    uint final_n = (((total < ef_val)) ? (total) : (ef_val));
    n_merged[0u] = final_n;
  }
}
