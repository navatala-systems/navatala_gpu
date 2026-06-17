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
extern "C" __global__ void navatala_vector_search_merge_sorted_f32(const float* a, const float* b, const unsigned int* a_indices, const unsigned int* b_indices, const unsigned int* n_a, const unsigned int* n_b, float* merged, unsigned int* merged_indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int na = n_a[0];
  unsigned int nb = n_b[0];
  unsigned int total = (na + nb);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int lo = 0u;
    unsigned int hi = gid;
    unsigned int hi_clamped = (((gid > na)) ? (na) : (gid));
    hi = hi_clamped;
    for (int __iter = 0; __iter < 32; ++__iter) {
      if (!((lo < hi))) break;
      unsigned int lo_v = lo;
      unsigned int hi_v = hi;
      unsigned int mid = (lo_v + ((hi_v - lo_v) / 2u));
      unsigned int j = (gid - mid);
      float a_val = (((mid < na)) ? (a[mid]) : (__uint_as_float(0x7e967699u)));
      float b_prev = (((j > 0u)) ? (b[(j - 1u)]) : (__uint_as_float(0xfe967699u)));
      bool go_higher = (a_val < b_prev);
      if (go_higher) {
        lo = (mid + 1u);
      } else {
        hi = mid;
      }
    }
    unsigned int i = lo;
    unsigned int j = (gid - i);
    bool a_available = (i < na);
    bool b_available = (j < nb);
    float a_val = ((a_available) ? (a[i]) : (__uint_as_float(0x7e967699u)));
    float b_val = ((b_available) ? (b[j]) : (__uint_as_float(0x7e967699u)));
    bool pick_a = (a_val <= b_val);
    float out_val = ((pick_a) ? (a_val) : (b_val));
    unsigned int out_idx = ((pick_a) ? (a_indices[i]) : (b_indices[j]));
    merged[gid] = out_val;
    merged_indices[gid] = out_idx;
  }
}
