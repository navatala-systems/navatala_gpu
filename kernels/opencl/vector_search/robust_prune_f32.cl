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

__kernel void navatala_vector_search_robust_prune_f32(__global const uint* vertex, __global const uint* candidates, __global const float* distances, __global const float* dataset, __global const uint* n_candidates, __global const uint* dim, __global const uint* max_degree, __global const uint* alpha, __global uint* selected, __global uint* n_selected) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint nc = n_candidates[0];
  uint md = max_degree[0];
  if (tid == (uint)(0u)) {
    uint limit = (((nc < md)) ? (nc) : (md));
    for (int i = 0; i < (int)(limit); ++i) {
      uint cand = candidates[i];
      selected[i] = cand;
    }
    n_selected[(uint)(0u)] = limit;
  }
}
