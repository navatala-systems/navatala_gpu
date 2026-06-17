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
extern "C" __global__ void navatala_vector_search_populate_inverted_lists_f32(const float* vectors, const unsigned int* labels, const unsigned int* list_offsets, const unsigned int* n_vectors, const unsigned int* dim, float* list_data, unsigned int* list_indices, unsigned int* write_positions) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int d = dim[0];
  bool inBounds = (gid < nv);
  if (inBounds) {
    unsigned int label = labels[gid];
    unsigned int list_base = list_offsets[label];
    unsigned int local_pos = atomicAdd(&(write_positions[label]), 1u);
    unsigned int global_pos = (list_base + local_pos);
    unsigned int src_base = (gid * d);
    unsigned int dst_base = (global_pos * d);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int src_off = (src_base + k);
      unsigned int dst_off = (dst_base + k);
      float val = vectors[src_off];
      list_data[dst_off] = val;
    }
    list_indices[global_pos] = gid;
  }
}
