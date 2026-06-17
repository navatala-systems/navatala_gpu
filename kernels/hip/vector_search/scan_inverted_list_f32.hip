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
extern "C" __global__ void navatala_vector_search_scan_inverted_list_f32(const float* queries, const float* list_data, const unsigned int* list_indices, const unsigned int* list_offsets, const unsigned int* probe_indices, const unsigned int* n_queries, const unsigned int* n_probes, const unsigned int* dim, const unsigned int* max_list_size, float* scan_distances, unsigned int* scan_indices, unsigned int* scan_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  unsigned int wg_size = 256u;
  unsigned int nq = n_queries[0];
  unsigned int np = n_probes[0];
  unsigned int d = dim[0];
  unsigned int mls = max_list_size[0];
  unsigned int total_pairs = (nq * np);
  bool valid_wg = (wgid < total_pairs);
  if (valid_wg) {
    unsigned int query_idx = (wgid / np);
    unsigned int probe_idx = (wgid % np);
    unsigned int probe_offset = ((query_idx * np) + probe_idx);
    unsigned int list_idx = probe_indices[probe_offset];
    unsigned int list_start = list_offsets[list_idx];
    unsigned int list_idx_plus1 = (list_idx + 1u);
    unsigned int list_end = list_offsets[list_idx_plus1];
    unsigned int list_size = (list_end - list_start);
    unsigned int query_base = (query_idx * d);
    unsigned int out_base = (((query_idx * np) * mls) + (probe_idx * mls));
    unsigned int iters = ((list_size / wg_size) + 1u);
    for (int iter = 0; iter < (int)(iters); ++iter) {
      unsigned int local_vec_idx = ((iter * wg_size) + tid);
      bool valid_vec = (local_vec_idx < list_size);
      if (valid_vec) {
        unsigned int global_vec_idx = (list_start + local_vec_idx);
        unsigned int vec_base = (global_vec_idx * d);
        float dist_acc = __uint_as_float(0x00000000u);
        for (int k = 0; k < (int)(d); ++k) {
          unsigned int q_off = (query_base + k);
          unsigned int v_off = (vec_base + k);
          float q_val = queries[q_off];
          float v_val = list_data[v_off];
          float diff = (q_val - v_val);
          float sq = (diff * diff);
          float old_d = dist_acc;
          dist_acc = (old_d + sq);
        }
        unsigned int orig_idx = list_indices[global_vec_idx];
        unsigned int out_offset = (out_base + local_vec_idx);
        float final_d = dist_acc;
        scan_distances[out_offset] = final_d;
        scan_indices[out_offset] = orig_idx;
      }
    }
    bool is_t0 = (tid == 0u);
    if (is_t0) {
      unsigned int count_offset = ((query_idx * np) + probe_idx);
      scan_counts[count_offset] = list_size;
    }
  }
}
