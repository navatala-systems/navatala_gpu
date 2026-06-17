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

__kernel void navatala_vector_search_normalize_p_q_codebook_f32(__global const int* codebook_sums_i32, __global const uint* codebook_counts, __global const uint* n_codewords, __global const uint* sub_dim, __global float* codebook) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint nc = n_codewords[0];
  uint sd = sub_dim[0];
  uint total = (nc * sd);
  float scale = as_float(0x47800000u);
  if ((tid < total)) {
    uint cw = (tid / sd);
    uint count_u32 = codebook_counts[cw];
    int sum_i32 = codebook_sums_i32[tid];
    float sum_f32 = (((float)(sum_i32)) / scale);
    float count_f32 = ((float)(count_u32));
    float _centroid = (((count_f32 > as_float(0x00000000u))) ? ((sum_f32 / count_f32)) : (as_float(0x00000000u)));
    codebook[tid] = _centroid;
  }
}
