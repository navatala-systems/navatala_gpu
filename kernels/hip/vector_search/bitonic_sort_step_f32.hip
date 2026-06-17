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
extern "C" __global__ void navatala_vector_search_bitonic_sort_step_f32(const unsigned int* n, const unsigned int* step, const unsigned int* stage, float* keys, unsigned int* values) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int stepVal = step[0];
  unsigned int stageVal = stage[0];
  unsigned int half_n = (nVal / 2u);
  bool inBounds = (gid < half_n);
  if (inBounds) {
    unsigned int distance = (1u << (stepVal - stageVal));
    unsigned int block_size = (2u << stepVal);
    unsigned int block_id = (gid / distance);
    unsigned int offset = (gid % distance);
    unsigned int i = ((block_id * (2u * distance)) + offset);
    unsigned int j = (i + distance);
    unsigned int block_start = ((i / block_size) * block_size);
    bool ascending = (((block_start / block_size) % 2u) == 0u);
    float key_i = keys[i];
    float key_j = keys[j];
    unsigned int val_i = values[i];
    unsigned int val_j = values[j];
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
