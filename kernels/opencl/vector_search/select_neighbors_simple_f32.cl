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

__kernel void navatala_vector_search_select_neighbors_simple_f32(__global const uint* candidates, __global const float* distances, __global const uint* n_candidates, __global const uint* m, __global uint* selected, __global uint* selected_count) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint n_cand = n_candidates[0];
  uint m_val = m[0];
  if (tid == (uint)(0u)) {
    uint limit = (((n_cand < m_val)) ? (n_cand) : (m_val));
    for (int i = 0; i < (int)(limit); ++i) {
      uint cand = candidates[i];
      selected[i] = cand;
    }
    selected_count[(uint)(0u)] = limit;
  }
}
