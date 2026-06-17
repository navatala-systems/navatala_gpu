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
extern "C" __global__ void navatala_vector_search_merge_search_results_f32(const unsigned int* ids_a, const float* dists_a, const unsigned int* ids_b, const float* dists_b, const unsigned int* n_queries, const unsigned int* k_a, const unsigned int* k_b, const unsigned int* k_out, unsigned int* merged_ids, float* merged_dists) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int query_id = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int ka = k_a[0];
  unsigned int kb = k_b[0];
  unsigned int ko = k_out[0];
  if ((query_id < nq)) {
    unsigned int ptr_a = 0u;
    unsigned int ptr_b = 0u;
    unsigned int out_idx = 0u;
    for (int _iter = 0; _iter < (int)(ko); ++_iter) {
      unsigned int pa = ptr_a;
      unsigned int pb = ptr_b;
      unsigned int oi = out_idx;
      bool a_valid = (pa < ka);
      bool b_valid = (pb < kb);
      unsigned int idx_a = ((query_id * ka) + pa);
      unsigned int idx_b = ((query_id * kb) + pb);
      float dist_a_raw = dists_a[idx_a];
      float dist_b_raw = dists_b[idx_b];
      float dist_a = ((a_valid) ? (dist_a_raw) : (__uint_as_float(0x7e967699u)));
      float dist_b = ((b_valid) ? (dist_b_raw) : (__uint_as_float(0x7e967699u)));
      bool choose_a = (dist_a <= dist_b);
      unsigned int chosen_id = ((choose_a) ? (ids_a[idx_a]) : (ids_b[idx_b]));
      float chosen_dist = ((choose_a) ? (dist_a) : (dist_b));
      unsigned int out_flat = ((query_id * ko) + oi);
      merged_ids[out_flat] = chosen_id;
      merged_dists[out_flat] = chosen_dist;
      if (choose_a) {
        ptr_a = (pa + 1u);
      } else {
        ptr_b = (pb + 1u);
      }
      out_idx = (oi + 1u);
    }
  }
}
