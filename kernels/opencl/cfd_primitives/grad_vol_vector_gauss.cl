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

__kernel void navatala_cfd_primitives_grad_vol_vector_gauss(__global const float* cellUx, __global const float* cellUy, __global const float* cellUz, __global const float* bcX, __global const float* bcY, __global const float* bcZ, __global const uint* bcMask, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const int* params, __global float* outXX, __global float* outXY, __global float* outXZ, __global float* outYX, __global float* outYY, __global float* outYZ, __global float* outZX, __global float* outZY, __global float* outZZ) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    float sumXX = as_float(0x00000000u);
    float sumXY = as_float(0x00000000u);
    float sumXZ = as_float(0x00000000u);
    float sumYX = as_float(0x00000000u);
    float sumYY = as_float(0x00000000u);
    float sumYZ = as_float(0x00000000u);
    float sumZX = as_float(0x00000000u);
    float sumZY = as_float(0x00000000u);
    float sumZZ = as_float(0x00000000u);
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
        float uxO = cellUx[o];
        float uyO = cellUy[o];
        float uzO = cellUz[o];
        float uxF = uxO;
        float uyF = uyO;
        float uzF = uzO;
        if ((f < params[2])) {
          int n = neighbour[f];
          float uxN = cellUx[n];
          float uyN = cellUy[n];
          float uzN = cellUz[n];
          float ww = weights[f];
          float iw = (as_float(0x3f800000u) - ww);
          uxF = ((ww * uxO) + (iw * uxN));
          uyF = ((ww * uyO) + (iw * uyN));
          uzF = ((ww * uzO) + (iw * uzN));
        } else {
          uint m = bcMask[f];
          if ((m == (uint)(1u))) {
            float ww = weights[f];
            float iw = (as_float(0x3f800000u) - ww);
            uxF = ((ww * uxO) + (iw * bcX[f]));
            uyF = ((ww * uyO) + (iw * bcY[f]));
            uzF = ((ww * uzO) + (iw * bcZ[f]));
          }
          if ((m == (uint)(2u))) {
            uxF = bcX[f];
            uyF = bcY[f];
            uzF = bcZ[f];
          }
        }
        float sfXf = sfX[f];
        float sfYf = sfY[f];
        float sfZf = sfZ[f];
        float wUx = (s * uxF);
        float wUy = (s * uyF);
        float wUz = (s * uzF);
        sumXX = (sumXX + (wUx * sfXf));
        sumXY = (sumXY + (wUx * sfYf));
        sumXZ = (sumXZ + (wUx * sfZf));
        sumYX = (sumYX + (wUy * sfXf));
        sumYY = (sumYY + (wUy * sfYf));
        sumYZ = (sumYZ + (wUy * sfZf));
        sumZX = (sumZX + (wUz * sfXf));
        sumZY = (sumZY + (wUz * sfYf));
        sumZZ = (sumZZ + (wUz * sfZf));
      }
    }
    float v = vol[((int)((int)(get_global_id(0))))];
    float invV = as_float(0x00000000u);
    if ((v != as_float(0x00000000u))) {
      invV = (as_float(0x3f800000u) / v);
    }
    outXX[((int)((int)(get_global_id(0))))] = (sumXX * invV);
    outXY[((int)((int)(get_global_id(0))))] = (sumXY * invV);
    outXZ[((int)((int)(get_global_id(0))))] = (sumXZ * invV);
    outYX[((int)((int)(get_global_id(0))))] = (sumYX * invV);
    outYY[((int)((int)(get_global_id(0))))] = (sumYY * invV);
    outYZ[((int)((int)(get_global_id(0))))] = (sumYZ * invV);
    outZX[((int)((int)(get_global_id(0))))] = (sumZX * invV);
    outZY[((int)((int)(get_global_id(0))))] = (sumZY * invV);
    outZZ[((int)((int)(get_global_id(0))))] = (sumZZ * invV);
  }
}
