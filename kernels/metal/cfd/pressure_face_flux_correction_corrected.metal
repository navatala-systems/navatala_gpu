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

kernel void navatala_cfd_pressure_face_flux_correction_corrected(device const int* owner [[buffer(0)]], device const int* neighbour [[buffer(1)]], device const float* weights [[buffer(2)]], device const float* gradX [[buffer(3)]], device const float* gradY [[buffer(4)]], device const float* gradZ [[buffer(5)]], device const float* magSf [[buffer(6)]], device const float* dCorrX [[buffer(7)]], device const float* dCorrY [[buffer(8)]], device const float* dCorrZ [[buffer(9)]], device const float* rAUf [[buffer(10)]], device const uint* params [[buffer(11)]], device float* outFaceFlux [[buffer(12)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(params[0]))) {
    return;
  } else {
    if (int(__gid.x) >= ((int)(params[1]))) {
      outFaceFlux[int(__gid.x)] = as_type<float>(0x00000000u);
    } else {
      int own = owner[int(__gid.x)];
      int nei = neighbour[int(__gid.x)];
      float w = weights[int(__gid.x)];
      float omw = (as_type<float>(0x3f800000u) - w);
      float gx = ((w * gradX[own]) + (omw * gradX[nei]));
      float gy = ((w * gradY[own]) + (omw * gradY[nei]));
      float gz = ((w * gradZ[own]) + (omw * gradZ[nei]));
      float corr = (((dCorrX[int(__gid.x)] * gx) + (dCorrY[int(__gid.x)] * gy)) + (dCorrZ[int(__gid.x)] * gz));
      outFaceFlux[int(__gid.x)] = (rAUf[int(__gid.x)] * (magSf[int(__gid.x)] * corr));
    }
  }
}
