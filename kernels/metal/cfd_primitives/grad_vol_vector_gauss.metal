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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_primitives_grad_vol_vector_gauss(device const float* cellUx [[buffer(0)]], device const float* cellUy [[buffer(1)]], device const float* cellUz [[buffer(2)]], device const float* bcX [[buffer(3)]], device const float* bcY [[buffer(4)]], device const float* bcZ [[buffer(5)]], device const uint* bcMask [[buffer(6)]], device const int* owner [[buffer(7)]], device const int* neighbour [[buffer(8)]], device const float* weights [[buffer(9)]], device const float* sfX [[buffer(10)]], device const float* sfY [[buffer(11)]], device const float* sfZ [[buffer(12)]], device const int* offsets [[buffer(13)]], device const int* faceIdx [[buffer(14)]], device const float* sign [[buffer(15)]], device const float* vol [[buffer(16)]], device const int* params [[buffer(17)]], device float* outXX [[buffer(18)]], device float* outXY [[buffer(19)]], device float* outXZ [[buffer(20)]], device float* outYX [[buffer(21)]], device float* outYY [[buffer(22)]], device float* outYZ [[buffer(23)]], device float* outZX [[buffer(24)]], device float* outZY [[buffer(25)]], device float* outZZ [[buffer(26)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
    float sumXX = as_type<float>(0x00000000u);
    float sumXY = as_type<float>(0x00000000u);
    float sumXZ = as_type<float>(0x00000000u);
    float sumYX = as_type<float>(0x00000000u);
    float sumYY = as_type<float>(0x00000000u);
    float sumYZ = as_type<float>(0x00000000u);
    float sumZX = as_type<float>(0x00000000u);
    float sumZY = as_type<float>(0x00000000u);
    float sumZZ = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < params[1]) {
        float s = sign[k];
        int o = owner[f];
        float uxO = cellUx[o];
        float uyO = cellUy[o];
        float uzO = cellUz[o];
        float uxF = uxO;
        float uyF = uyO;
        float uzF = uzO;
        if (f < params[2]) {
          int n = neighbour[f];
          float uxN = cellUx[n];
          float uyN = cellUy[n];
          float uzN = cellUz[n];
          float ww = weights[f];
          float iw = (as_type<float>(0x3f800000u) - ww);
          uxF = ((ww * uxO) + (iw * uxN));
          uyF = ((ww * uyO) + (iw * uyN));
          uzF = ((ww * uzO) + (iw * uzN));
        } else {
          uint m = bcMask[f];
          if (m == 1u) {
            float ww = weights[f];
            float iw = (as_type<float>(0x3f800000u) - ww);
            uxF = ((ww * uxO) + (iw * bcX[f]));
            uyF = ((ww * uyO) + (iw * bcY[f]));
            uzF = ((ww * uzO) + (iw * bcZ[f]));
          }
          if (m == 2u) {
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
    float v = vol[((int)(int(__gid.x)))];
    float invV = as_type<float>(0x00000000u);
    if (v != as_type<float>(0x00000000u)) {
      invV = (as_type<float>(0x3f800000u) / v);
    }
    outXX[((int)(int(__gid.x)))] = (sumXX * invV);
    outXY[((int)(int(__gid.x)))] = (sumXY * invV);
    outXZ[((int)(int(__gid.x)))] = (sumXZ * invV);
    outYX[((int)(int(__gid.x)))] = (sumYX * invV);
    outYY[((int)(int(__gid.x)))] = (sumYY * invV);
    outYZ[((int)(int(__gid.x)))] = (sumYZ * invV);
    outZX[((int)(int(__gid.x)))] = (sumZX * invV);
    outZY[((int)(int(__gid.x)))] = (sumZY * invV);
    outZZ[((int)(int(__gid.x)))] = (sumZZ * invV);
  }
}
