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

__kernel void navatala_cfd_exact_mat_vec(__global const float* x, __global const int* owner, __global const int* neighbour, __global const uint* offsets, __global const uint* faceIdx, __global const int* sign, __global const float* upper, __global const float* lower, __global const float* diag, __global const int* counts, __global float* outAx) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    float s = (diag[(int)(get_global_id(0))] * x[(int)(get_global_id(0))]);
    uint beg = offsets[(int)(get_global_id(0))];
    int c1 = ((int)(get_global_id(0)) + 1);
    uint end = offsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      uint f = faceIdx[k];
      if (((int)(f)) < ((int)(counts[2]))) {
        int sg = sign[k];
        if (sg >= 0) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * x[nb]));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * x[ow]));
        }
      }
    }
    outAx[(int)(get_global_id(0))] = s;
  }
}
