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

kernel void navatala_cfd_phi_correct(device const float* phiIn [[buffer(0)]], device const float* pCell [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* neighbour [[buffer(3)]], device const float* magSf [[buffer(4)]], device const float* deltaCoeffs [[buffer(5)]], device const float* coeff [[buffer(6)]], device const float* bcVal [[buffer(7)]], device const int* bcMask [[buffer(8)]], device const float* bcSnGrad [[buffer(9)]], device const int* bcSnGradMask [[buffer(10)]], device const float* faceFluxCorrection [[buffer(11)]], device const int* counts [[buffer(12)]], device const float* paramsF [[buffer(13)]], device float* phiOut [[buffer(14)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float po = pCell[o];
    float other = po;
    float snGrad = as_type<float>(0x00000000u);
    if (((int)(int(__gid.x))) < counts[1]) {
      int n = neighbour[((int)(int(__gid.x)))];
      other = pCell[n];
    } else {
      if (bcMask[((int)(int(__gid.x)))] != 0) {
        other = bcVal[((int)(int(__gid.x)))];
      }
    }
    snGrad = (deltaCoeffs[((int)(int(__gid.x)))] * (other - po));
    if (bcSnGradMask[((int)(int(__gid.x)))] != 0) {
      snGrad = bcSnGrad[((int)(int(__gid.x)))];
    }
    float gradFace = (magSf[((int)(int(__gid.x)))] * snGrad);
    phiOut[((int)(int(__gid.x)))] = (phiIn[((int)(int(__gid.x)))] - ((paramsF[0] * (coeff[((int)(int(__gid.x)))] * gradFace)) + faceFluxCorrection[((int)(int(__gid.x)))]));
  }
}
