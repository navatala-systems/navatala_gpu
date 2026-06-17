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
extern "C" __global__ void navatala_vector_search_jaccard_distance_u32(const unsigned int* x, const unsigned int* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim_words, float* dist) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int dw = dim_words[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    unsigned int intersection = 0u;
    unsigned int union_count = 0u;
    for (int k = 0; k < (int)(dw); ++k) {
      unsigned int x_idx = ((gid_x * dw) + k);
      unsigned int y_idx = ((gid_y * dw) + k);
      unsigned int x_val = x[x_idx];
      unsigned int y_val = y[y_idx];
      unsigned int inter = (x_val & y_val);
      unsigned int uni = (x_val | y_val);
      unsigned int i0 = inter;
      unsigned int m1 = 1431655765u;
      unsigned int m2 = 858993459u;
      unsigned int m4 = 252645135u;
      unsigned int i1 = (i0 - ((i0 >> 1u) & m1));
      unsigned int i2 = ((i1 & m2) + ((i1 >> 2u) & m2));
      unsigned int i3 = ((i2 + (i2 >> 4u)) & m4);
      unsigned int i4 = (i3 + (i3 >> 8u));
      unsigned int i5 = (i4 + (i4 >> 16u));
      unsigned int pop_inter = (i5 & 63u);
      unsigned int u0 = uni;
      unsigned int u1 = (u0 - ((u0 >> 1u) & m1));
      unsigned int u2 = ((u1 & m2) + ((u1 >> 2u) & m2));
      unsigned int u3 = ((u2 + (u2 >> 4u)) & m4);
      unsigned int u4 = (u3 + (u3 >> 8u));
      unsigned int u5 = (u4 + (u4 >> 16u));
      unsigned int pop_union = (u5 & 63u);
      unsigned int old_inter = intersection;
      unsigned int old_union = union_count;
      intersection = (old_inter + pop_inter);
      union_count = (old_union + pop_union);
    }
    unsigned int final_inter = intersection;
    unsigned int final_union = union_count;
    float inter_f = ((float)(final_inter));
    float union_f = ((float)(final_union));
    float eps = __uint_as_float(0x322bcc77u);
    float safe_union = (union_f + eps);
    float similarity = (inter_f / safe_union);
    float jaccard_dist = (__uint_as_float(0x3f800000u) - similarity);
    unsigned int out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = jaccard_dist;
  }
}
