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

__kernel void navatala_cfd_primitives_grad_vol_scalar_gauss(__global const float* cellPhi, __global const float* bcValue, __global const uint* bcMask, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const int* params, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    float sumX = as_float(0x00000000u);
    float sumY = as_float(0x00000000u);
    float sumZ = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if ((f < params[1])) {
        float s = sign[k];
        int o = owner[f];
        float po = cellPhi[o];
        float phiF = po;
        if ((f < params[2])) {
          int n = neighbour[f];
          float pn = cellPhi[n];
          float ww = weights[f];
          float iw = (as_float(0x3f800000u) - ww);
          phiF = ((ww * po) + (iw * pn));
        } else {
          uint m = bcMask[f];
          if ((m == (uint)(1u))) {
            float ww = weights[f];
            float iw = (as_float(0x3f800000u) - ww);
            phiF = ((ww * po) + (iw * bcValue[f]));
          }
          if ((m == (uint)(2u))) {
            phiF = bcValue[f];
          }
        }
        float w = (s * phiF);
        sumX = (sumX + (w * sfX[f]));
        sumY = (sumY + (w * sfY[f]));
        sumZ = (sumZ + (w * sfZ[f]));
      }
    }
    float v = vol[((int)((int)(get_global_id(0))))];
    float invV = as_float(0x00000000u);
    if ((v != as_float(0x00000000u))) {
      invV = (as_float(0x3f800000u) / v);
    }
    outX[((int)((int)(get_global_id(0))))] = (sumX * invV);
    outY[((int)((int)(get_global_id(0))))] = (sumY * invV);
    outZ[((int)((int)(get_global_id(0))))] = (sumZ * invV);
  }
}
