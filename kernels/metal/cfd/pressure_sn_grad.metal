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

kernel void navatala_cfd_pressure_sn_grad(device const float* phiHbyA [[buffer(0)]], device const int* ownerAll [[buffer(1)]], device const float* sfX [[buffer(2)]], device const float* sfY [[buffer(3)]], device const float* sfZ [[buffer(4)]], device const float* magSf [[buffer(5)]], device const float* ux [[buffer(6)]], device const float* uy [[buffer(7)]], device const float* uz [[buffer(8)]], device const float* uBcX [[buffer(9)]], device const float* uBcY [[buffer(10)]], device const float* uBcZ [[buffer(11)]], device const uint* uBcMask [[buffer(12)]], device const float* rAUf [[buffer(13)]], device const uint* snGradMask [[buffer(14)]], device const uint* params [[buffer(15)]], device float* outSnGrad [[buffer(16)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(params[1]))) {
    return;
  } else {
    uint mask = snGradMask[int(__gid.x)];
    if (((int)(mask)) == 0) {
      outSnGrad[int(__gid.x)] = as_type<float>(0x00000000u);
    } else {
      int o = ownerAll[int(__gid.x)];
      if (o < 0) {
        outSnGrad[int(__gid.x)] = as_type<float>(0x00000000u);
      } else {
        if (o >= ((int)(params[0]))) {
          outSnGrad[int(__gid.x)] = as_type<float>(0x00000000u);
        } else {
          float uxF = ux[o];
          float uyF = uy[o];
          float uzF = uz[o];
          uint bcm = uBcMask[int(__gid.x)];
          if (((int)(bcm)) == 2) {
            uxF = uBcX[int(__gid.x)];
            uyF = uBcY[int(__gid.x)];
            uzF = uBcZ[int(__gid.x)];
          }
          float phiU = (((sfX[int(__gid.x)] * uxF) + (sfY[int(__gid.x)] * uyF)) + (sfZ[int(__gid.x)] * uzF));
          float rf = rAUf[int(__gid.x)];
          float ms = magSf[int(__gid.x)];
          float denom = (rf * ms);
          float ph = phiHbyA[int(__gid.x)];
          if (abs(denom) < as_type<float>(0x1e3ce508u)) {
            outSnGrad[int(__gid.x)] = as_type<float>(0x00000000u);
          } else {
            outSnGrad[int(__gid.x)] = ((ph - (rf * phiU)) / denom);
          }
        }
      }
    }
  }
}
