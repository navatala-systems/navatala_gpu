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

kernel void navatala_cfd_k_omega_s_s_t_nut_compute(device const float* kVals [[buffer(0)]], device const float* omegaVals [[buffer(1)]], device const float* f23Vals [[buffer(2)]], device const float* s2Vals [[buffer(3)]], device const uint* counts [[buffer(4)]], device const float* params [[buffer(5)]], device float* outNut [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float a1 = params[0];
    float b1 = params[1];
    float k = kVals[int(__gid.x)];
    float om = omegaVals[int(__gid.x)];
    float f23 = f23Vals[int(__gid.x)];
    float s2raw = s2Vals[int(__gid.x)];
    float denomA = (a1 * om);
    float denomB = ((b1 * f23) * sqrt((s2raw * ((float)((s2raw > as_type<float>(0x00000000u)))))));
    float num = (a1 * k);
    if (denomA > denomB) {
      if (denomA > as_type<float>(0x00000000u)) {
        outNut[int(__gid.x)] = (num / denomA);
      } else {
        outNut[int(__gid.x)] = as_type<float>(0x00000000u);
      }
    } else {
      if (denomB > as_type<float>(0x00000000u)) {
        outNut[int(__gid.x)] = (num / denomB);
      } else {
        outNut[int(__gid.x)] = as_type<float>(0x00000000u);
      }
    }
  }
}
