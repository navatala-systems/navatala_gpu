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
extern "C" __global__ void navatala_vector_search_merge_probe_results_f32(const float* scan_distances, const unsigned int* scan_indices, const unsigned int* scan_counts, const unsigned int* n_queries, const unsigned int* n_probes, const unsigned int* max_list_size, const unsigned int* k, float* final_distances, unsigned int* final_indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int query_idx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int wg_size = 256u;
  unsigned int nq = n_queries[0];
  unsigned int np = n_probes[0];
  unsigned int mls = max_list_size[0];
  unsigned int kval = k[0];
  bool valid_query = (query_idx < nq);
  if (valid_query) {
    extern __shared__ float s_best_dists[]; /* dynamic shared memory (best-effort) */
    extern __shared__ unsigned int s_best_idxs[]; /* dynamic shared memory (best-effort) */
    s_best_dists[tid] = __uint_as_float(0x7f7fc99eu);
    s_best_idxs[tid] = 0u;
    unsigned int total_max = (np * mls);
    unsigned int iters = ((total_max / wg_size) + 1u);
    float my_best_dist = __uint_as_float(0x7f7fc99eu);
    unsigned int my_best_idx = 0u;
    for (int iter = 0; iter < (int)(iters); ++iter) {
      unsigned int flat_idx = ((iter * wg_size) + tid);
      bool valid_idx = (flat_idx < total_max);
      if (valid_idx) {
        unsigned int probe_idx = (flat_idx / mls);
        unsigned int local_vec = (flat_idx % mls);
        unsigned int count_offset = ((query_idx * np) + probe_idx);
        unsigned int actual_count = scan_counts[count_offset];
        bool within_count = (local_vec < actual_count);
        if (within_count) {
          unsigned int scan_offset = (((query_idx * np) * mls) + flat_idx);
          float dist = scan_distances[scan_offset];
          unsigned int idx = scan_indices[scan_offset];
          float cur_best = my_best_dist;
          bool better = (dist < cur_best);
          if (better) {
            my_best_dist = dist;
            my_best_idx = idx;
          }
        }
      }
    }
    float my_d = my_best_dist;
    unsigned int my_i = my_best_idx;
    s_best_dists[tid] = my_d;
    s_best_idxs[tid] = my_i;
    __syncthreads();
    bool is_t0 = (tid == 0u);
    if (is_t0) {
      for (int ki = 0; ki < (int)(kval); ++ki) {
        float sel_dist = __uint_as_float(0x7f7fc99eu);
        unsigned int sel_idx = 0u;
        unsigned int sel_slot = 0u;
        for (int s = 0; s < (int)(wg_size); ++s) {
          float sd = s_best_dists[s];
          unsigned int si = s_best_idxs[s];
          float cur_sel = sel_dist;
          bool b = (sd < cur_sel);
          if (b) {
            sel_dist = sd;
            sel_idx = si;
            sel_slot = s;
          }
        }
        unsigned int out_off = ((query_idx * kval) + ki);
        unsigned int final_sel_idx = sel_idx;
        float final_sel_dist = sel_dist;
        final_distances[out_off] = final_sel_dist;
        final_indices[out_off] = final_sel_idx;
        unsigned int slot_mark = sel_slot;
        s_best_dists[slot_mark] = __uint_as_float(0x7f7fc99eu);
      }
    }
    __syncthreads();
  }
}
