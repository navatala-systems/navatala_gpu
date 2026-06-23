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

__kernel void navatala_cfd_grad_alpha_cell(__global const float* alphaF, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const int* counts, __global float* gx, __global float* gy, __global float* gz) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float sx = as_float(0x00000000u);
    float sy = as_float(0x00000000u);
    float sz = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = (sign[k] * alphaF[f]);
      sx = (sx + (s * sfX[f]));
      sy = (sy + (s * sfY[f]));
      sz = (sz + (s * sfZ[f]));
    }
    float invV = (as_float(0x3f800000u) / vol[((int)((int)(get_global_id(0))))]);
    gx[((int)((int)(get_global_id(0))))] = (sx * invV);
    gy[((int)((int)(get_global_id(0))))] = (sy * invV);
    gz[((int)((int)(get_global_id(0))))] = (sz * invV);
  }
}
