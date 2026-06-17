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

__kernel void navatala_vector_search_update_candidate_list_f32(__global uint* candidates, __global float* distances, __global const uint* new_candidates, __global const float* new_distances, __global const uint* n_current, __global const uint* n_new, __global const uint* ef, __global uint* n_merged) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint n_curr = n_current[0];
  uint n_new_val = n_new[0];
  uint ef_val = ef[0];
  __local uint merge_ids[512];
  __local float merge_dists[512];
  if ((tid < n_curr)) {
    uint cid = candidates[tid];
    float cdist = distances[tid];
    merge_ids[tid] = cid;
    merge_dists[tid] = cdist;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((tid < n_new_val)) {
    uint offset = (n_curr + tid);
    uint nid = new_candidates[tid];
    float ndist = new_distances[tid];
    merge_ids[offset] = nid;
    merge_dists[offset] = ndist;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((tid == (uint)(0u))) {
    uint total = (n_curr + n_new_val);
    uint final_n = (((total < ef_val)) ? (total) : (ef_val));
    n_merged[(uint)(0u)] = final_n;
  }
}
