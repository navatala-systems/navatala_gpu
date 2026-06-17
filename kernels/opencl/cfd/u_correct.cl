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

__kernel void navatala_cfd_u_correct(__global const float* pCell, __global const int* owner, __global const int* neighbour, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* bcVal, __global const int* bcMask, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const float* rAU, __global const float* ux, __global const float* uy, __global const float* uz, __global const int* counts, __global const float* paramsF, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float gx = as_float(0x00000000u);
    float gy = as_float(0x00000000u);
    float gz = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      int o = owner[f];
      float pf = pCell[o];
      if ((f < counts[2])) {
        int n = neighbour[f];
        pf = (as_float(0x3f000000u) * (pCell[o] + pCell[n]));
      } else {
        if ((bcMask[f] != 0)) {
          pf = bcVal[f];
        }
      }
      float s = sign[k];
      gx = (gx + (s * (sfX[f] * pf)));
      gy = (gy + (s * (sfY[f] * pf)));
      gz = (gz + (s * (sfZ[f] * pf)));
    }
    float invV = (as_float(0x3f800000u) / vol[((int)((int)(get_global_id(0))))]);
    float sC = (paramsF[0] * rAU[((int)((int)(get_global_id(0))))]);
    float dUx = ((gx * invV) * sC);
    float dUy = ((gy * invV) * sC);
    float dUz = ((gz * invV) * sC);
    outX[((int)((int)(get_global_id(0))))] = (ux[((int)((int)(get_global_id(0))))] - dUx);
    outY[((int)((int)(get_global_id(0))))] = (uy[((int)((int)(get_global_id(0))))] - dUy);
    outZ[((int)((int)(get_global_id(0))))] = (uz[((int)((int)(get_global_id(0))))] - dUz);
  }
}
