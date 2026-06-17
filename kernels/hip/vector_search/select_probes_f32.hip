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
extern "C" __global__ void navatala_vector_search_select_probes_f32(const float* centroid_distances, const unsigned int* n_queries, const unsigned int* n_lists, const unsigned int* n_probes, unsigned int* probe_indices, float* probe_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int query_idx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int wg_size = 256u;
  unsigned int nq = n_queries[0];
  unsigned int nl = n_lists[0];
  unsigned int np = n_probes[0];
  bool query_valid = (query_idx < nq);
  if (query_valid) {
    extern __shared__ float s_distances[]; /* dynamic shared memory (best-effort) */
    extern __shared__ unsigned int s_indices[]; /* dynamic shared memory (best-effort) */
    s_distances[tid] = __uint_as_float(0x7f7fc99eu);
    s_indices[tid] = 0u;
    unsigned int base = (query_idx * nl);
    float local_min_dist = __uint_as_float(0x7f7fc99eu);
    unsigned int local_min_idx = 0u;
    unsigned int iters = ((nl / wg_size) + 1u);
    for (int iter = 0; iter < (int)(iters); ++iter) {
      unsigned int list_idx = ((iter * wg_size) + tid);
      bool valid = (list_idx < nl);
      if (valid) {
        unsigned int offset = (base + list_idx);
        float dist = centroid_distances[offset];
        float cur_min = local_min_dist;
        bool is_better = (dist < cur_min);
        if (is_better) {
          local_min_dist = dist;
          local_min_idx = list_idx;
        }
      }
    }
    float final_min_dist = local_min_dist;
    unsigned int final_min_idx = local_min_idx;
    s_distances[tid] = final_min_dist;
    s_indices[tid] = final_min_idx;
    __syncthreads();
    bool is_thread0 = (tid == 0u);
    if (is_thread0) {
      for (int p = 0; p < (int)(np); ++p) {
        float best_dist = __uint_as_float(0x7f7fc99eu);
        unsigned int best_idx = 0u;
        unsigned int best_slot = 0u;
        for (int s = 0; s < (int)(wg_size); ++s) {
          float curr_dist = s_distances[s];
          unsigned int curr_idx = s_indices[s];
          float cur_best = best_dist;
          bool better = (curr_dist < cur_best);
          if (better) {
            best_dist = curr_dist;
            best_idx = curr_idx;
            best_slot = s;
          }
        }
        unsigned int out_base = (query_idx * np);
        unsigned int out_offset = (out_base + p);
        unsigned int sel_idx = best_idx;
        float sel_dist = best_dist;
        probe_indices[out_offset] = sel_idx;
        probe_distances[out_offset] = sel_dist;
        unsigned int slot_to_mark = best_slot;
        s_distances[slot_to_mark] = __uint_as_float(0x7f7fc99eu);
      }
    }
    __syncthreads();
  }
}
