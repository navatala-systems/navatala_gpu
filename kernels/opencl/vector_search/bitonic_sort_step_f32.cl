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

__kernel void navatala_vector_search_bitonic_sort_step_f32(__global const uint* n, __global const uint* step, __global const uint* stage, __global float* keys, __global uint* values) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint stepVal = step[0];
  uint stageVal = stage[0];
  uint half_n = (nVal / (uint)(2u));
  bool inBounds = (gid < half_n);
  if (inBounds) {
    uint distance = ((uint)(1u) << (stepVal - stageVal));
    uint block_size = ((uint)(2u) << stepVal);
    uint block_id = (gid / distance);
    uint offset = (gid % distance);
    uint i = ((block_id * ((uint)(2u) * distance)) + offset);
    uint j = (i + distance);
    uint block_start = ((i / block_size) * block_size);
    bool ascending = (((block_start / block_size) % (uint)(2u)) == (uint)(0u));
    float key_i = keys[i];
    float key_j = keys[j];
    uint val_i = values[i];
    uint val_j = values[j];
    bool should_swap_asc = (key_i > key_j);
    bool should_swap_desc = (key_i < key_j);
    bool should_swap = ((ascending) ? (should_swap_asc) : (should_swap_desc));
    if (should_swap) {
      keys[i] = key_j;
      keys[j] = key_i;
      values[i] = val_j;
      values[j] = val_i;
    }
  }
}
