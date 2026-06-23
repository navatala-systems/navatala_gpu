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

kernel void navatala_cfd_primitives_rho_phi_from_rho_and_phi(device const float* phiAll [[buffer(0)]], device const float* rhoCell [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* neighbour [[buffer(3)]], device const float* weights [[buffer(4)]], device const float* rhoBcVal [[buffer(5)]], device const uint* rhoBcMask [[buffer(6)]], device const int* params [[buffer(7)]], device float* outRhoPhi [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float rhoF = rhoCell[o];
    if (((int)(int(__gid.x))) < params[1]) {
      int n = neighbour[((int)(int(__gid.x)))];
      float w0 = weights[((int)(int(__gid.x)))];
      float iw = (as_type<float>(0x3f800000u) - w0);
      rhoF = ((w0 * rhoCell[o]) + (iw * rhoCell[n]));
    } else {
      uint m = rhoBcMask[((int)(int(__gid.x)))];
      if (m == 1u) {
        float w0 = weights[((int)(int(__gid.x)))];
        float iw = (as_type<float>(0x3f800000u) - w0);
        rhoF = ((w0 * rhoCell[o]) + (iw * rhoBcVal[((int)(int(__gid.x)))]));
      }
      if (m == 2u) {
        rhoF = rhoBcVal[((int)(int(__gid.x)))];
      }
    }
    float flux = phiAll[((int)(int(__gid.x)))];
    outRhoPhi[((int)(int(__gid.x)))] = (rhoF * flux);
  }
}
