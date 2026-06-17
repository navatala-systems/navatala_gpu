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

kernel void navatala_cfd_primitives_interp_vector_face(device const int* owner [[buffer(0)]], device const int* neighbour [[buffer(1)]], device const float* weights [[buffer(2)]], device const float* cellX [[buffer(3)]], device const float* cellY [[buffer(4)]], device const float* cellZ [[buffer(5)]], device const float* bcX [[buffer(6)]], device const float* bcY [[buffer(7)]], device const float* bcZ [[buffer(8)]], device const uint* bcMask [[buffer(9)]], device const int* params [[buffer(10)]], device float* outX [[buffer(11)]], device float* outY [[buffer(12)]], device float* outZ [[buffer(13)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float uxo = cellX[o];
    float uyo = cellY[o];
    float uzo = cellZ[o];
    if ((((int)(int(__gid.x))) < params[1])) {
      int n = neighbour[((int)(int(__gid.x)))];
      float uxn = cellX[n];
      float uyn = cellY[n];
      float uzn = cellZ[n];
      float w = weights[((int)(int(__gid.x)))];
      float iw = (as_type<float>(0x3f800000u) - w);
      outX[((int)(int(__gid.x)))] = ((w * uxo) + (iw * uxn));
      outY[((int)(int(__gid.x)))] = ((w * uyo) + (iw * uyn));
      outZ[((int)(int(__gid.x)))] = ((w * uzo) + (iw * uzn));
    } else {
      uint m = bcMask[((int)(int(__gid.x)))];
      if ((m == 1u)) {
        float w = weights[((int)(int(__gid.x)))];
        float iw = (as_type<float>(0x3f800000u) - w);
        outX[((int)(int(__gid.x)))] = ((w * uxo) + (iw * bcX[((int)(int(__gid.x)))]));
        outY[((int)(int(__gid.x)))] = ((w * uyo) + (iw * bcY[((int)(int(__gid.x)))]));
        outZ[((int)(int(__gid.x)))] = ((w * uzo) + (iw * bcZ[((int)(int(__gid.x)))]));
      } else {
        if ((m == 2u)) {
          outX[((int)(int(__gid.x)))] = bcX[((int)(int(__gid.x)))];
          outY[((int)(int(__gid.x)))] = bcY[((int)(int(__gid.x)))];
          outZ[((int)(int(__gid.x)))] = bcZ[((int)(int(__gid.x)))];
        } else {
          outX[((int)(int(__gid.x)))] = uxo;
          outY[((int)(int(__gid.x)))] = uyo;
          outZ[((int)(int(__gid.x)))] = uzo;
        }
      }
    }
  }
}
