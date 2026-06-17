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

kernel void navatala_cfd_face_flux(device const float* x [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* neighbour [[buffer(2)]], device const float* cf [[buffer(3)]], device const float* bcVal [[buffer(4)]], device const int* bcMask [[buffer(5)]], device const int* counts [[buffer(6)]], device float* outFlux [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts[1])) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float xo = x[o];
    float other = xo;
    if ((((int)(int(__gid.x))) < counts[2])) {
      int n = neighbour[((int)(int(__gid.x)))];
      other = x[n];
    } else {
      int m = bcMask[((int)(int(__gid.x)))];
      if (((m == 1) || (m == 3))) {
        other = bcVal[((int)(int(__gid.x)))];
      } else {
        if ((m == 2)) {
          other = as_type<float>(0x00000000u);
        } else {
          if ((m == 4)) {
            outFlux[((int)(int(__gid.x)))] = bcVal[((int)(int(__gid.x)))];
            return;
          }
        }
      }
    }
    outFlux[((int)(int(__gid.x)))] = (cf[((int)(int(__gid.x)))] * (other - xo));
  }
}
