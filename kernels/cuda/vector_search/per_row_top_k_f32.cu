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
extern "C" __global__ void navatala_vector_search_per_row_top_k_f32(const float* data, const unsigned int* n_rows, const unsigned int* row_size, const unsigned int* k, float* top_values, unsigned int* top_indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nr = n_rows[0];
  unsigned int rs = row_size[0];
  unsigned int kVal = k[0];
  unsigned int total = (nr * kVal);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int row_idx = (gid / kVal);
    unsigned int k_idx = (gid % kVal);
    unsigned int row_start = (row_idx * rs);
    float best_val = __uint_as_float(0x7e967699u);
    unsigned int best_idx = 0u;
    unsigned int count_smaller = 0u;
    for (int i = 0; i < (int)(rs); ++i) {
      unsigned int data_idx = (row_start + i);
      float val = data[data_idx];
      unsigned int smaller_count = 0u;
      for (int j = 0; j < (int)(rs); ++j) {
        unsigned int other_idx = (row_start + j);
        float other_val = data[other_idx];
        bool is_smaller = (other_val < val);
        bool is_equal_earlier = ((other_val == val) && (j < i));
        bool counts = (is_smaller || is_equal_earlier);
        unsigned int old_cnt = smaller_count;
        unsigned int incr = ((counts) ? (1u) : (0u));
        smaller_count = (old_cnt + incr);
      }
      unsigned int cnt = smaller_count;
      bool is_kth = (cnt == k_idx);
      if (is_kth) {
        best_val = val;
        best_idx = i;
      }
    }
    unsigned int out_idx = ((row_idx * kVal) + k_idx);
    float final_val = best_val;
    unsigned int final_idx = best_idx;
    top_values[out_idx] = final_val;
    top_indices[out_idx] = final_idx;
  }
}
