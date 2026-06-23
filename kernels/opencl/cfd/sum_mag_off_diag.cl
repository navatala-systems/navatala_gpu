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

__kernel void navatala_cfd_sum_mag_off_diag(__global const float* upper, __global const float* lower, __global const int* offsets, __global const int* faceIdx, __global const int* sign, __global const int* counts, __global float* outSumOff) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float sum = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if ((f < counts[1])) {
        int sgn = sign[k];
        if ((sgn > 0)) {
          float val = upper[f];
          float negVal = (as_float(0x00000000u) - val);
          float absVal = (((val >= as_float(0x00000000u))) ? (val) : (negVal));
          sum = (sum + absVal);
        } else {
          float val = lower[f];
          float negVal = (as_float(0x00000000u) - val);
          float absVal = (((val >= as_float(0x00000000u))) ? (val) : (negVal));
          sum = (sum + absVal);
        }
      }
    }
    outSumOff[((int)((int)(get_global_id(0))))] = sum;
  }
}
