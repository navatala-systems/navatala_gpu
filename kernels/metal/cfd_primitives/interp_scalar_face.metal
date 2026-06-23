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

kernel void navatala_cfd_primitives_interp_scalar_face(device const int* owner [[buffer(0)]], device const int* neighbour [[buffer(1)]], device const float* weights [[buffer(2)]], device const float* cellPhi [[buffer(3)]], device const float* bcValue [[buffer(4)]], device const uint* bcMask [[buffer(5)]], device const int* params [[buffer(6)]], device float* outPhiF [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float po = cellPhi[o];
    if (((int)(int(__gid.x))) < params[1]) {
      int n = neighbour[((int)(int(__gid.x)))];
      float pn = cellPhi[n];
      float w = weights[((int)(int(__gid.x)))];
      float iw = (as_type<float>(0x3f800000u) - w);
      outPhiF[((int)(int(__gid.x)))] = ((w * po) + (iw * pn));
    } else {
      if (bcMask[((int)(int(__gid.x)))] == 1u) {
        float w = weights[((int)(int(__gid.x)))];
        float iw = (as_type<float>(0x3f800000u) - w);
        outPhiF[((int)(int(__gid.x)))] = ((w * po) + (iw * bcValue[((int)(int(__gid.x)))]));
      } else {
        if (bcMask[((int)(int(__gid.x)))] == 2u) {
          outPhiF[((int)(int(__gid.x)))] = bcValue[((int)(int(__gid.x)))];
        } else {
          outPhiF[((int)(int(__gid.x)))] = po;
        }
      }
    }
  }
}
