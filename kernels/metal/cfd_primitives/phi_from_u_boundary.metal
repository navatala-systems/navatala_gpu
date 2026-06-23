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

kernel void navatala_cfd_primitives_phi_from_u_boundary(device const int* owner [[buffer(0)]], device const float* weights [[buffer(1)]], device const float* sfX [[buffer(2)]], device const float* sfY [[buffer(3)]], device const float* sfZ [[buffer(4)]], device const float* ux [[buffer(5)]], device const float* uy [[buffer(6)]], device const float* uz [[buffer(7)]], device const float* bcx [[buffer(8)]], device const float* bcy [[buffer(9)]], device const float* bcz [[buffer(10)]], device const uint* bcmask [[buffer(11)]], device const uint* params [[buffer(12)]], device float* outPhi [[buffer(13)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int f = (((int)(params[1])) + int(__gid.x));
  if (f >= ((int)(params[0]))) {
    return;
  } else {
    int o = owner[f];
    float uox = ux[o];
    float uoy = uy[o];
    float uoz = uz[o];
    float ufx = uox;
    float ufy = uoy;
    float ufz = uoz;
    uint m = bcmask[f];
    int mInt = ((int)(m));
    if (mInt == 1) {
      float w = weights[f];
      float iw = (as_type<float>(0x3f800000u) - w);
      ufx = ((w * uox) + (iw * bcx[f]));
      ufy = ((w * uoy) + (iw * bcy[f]));
      ufz = ((w * uoz) + (iw * bcz[f]));
    } else {
      if (mInt == 2) {
        ufx = bcx[f];
        ufy = bcy[f];
        ufz = bcz[f];
      }
    }
    outPhi[f] = (((sfX[f] * ufx) + (sfY[f] * ufy)) + (sfZ[f] * ufz));
  }
}
