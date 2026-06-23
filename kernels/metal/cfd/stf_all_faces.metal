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

kernel void navatala_cfd_stf_all_faces(device const float* divN [[buffer(0)]], device const float* alpha [[buffer(1)]], device const float* alphaF [[buffer(2)]], device const float* delta [[buffer(3)]], device const int* owner [[buffer(4)]], device const int* nei [[buffer(5)]], device const float* weights [[buffer(6)]], device const float* contactGrad [[buffer(7)]], device const uint* thetaMask [[buffer(8)]], device const float* paramsF [[buffer(9)]], device const int* counts [[buffer(10)]], device float* stf [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    int n = nei[((int)(int(__gid.x)))];
    float w = weights[((int)(int(__gid.x)))];
    float wm = (as_type<float>(0x3f800000u) - w);
    float divInterp = ((w * divN[o]) + (wm * divN[n]));
    float Kf = (as_type<float>(0xbf800000u) * divInterp);
    float snGrad = as_type<float>(0x00000000u);
    if (((int)(int(__gid.x))) < counts[2]) {
      snGrad = (delta[((int)(int(__gid.x)))] * (alpha[n] - alpha[o]));
    } else {
      if (thetaMask[((int)(int(__gid.x)))] != 0u) {
        snGrad = contactGrad[((int)(int(__gid.x)))];
      } else {
        snGrad = (delta[((int)(int(__gid.x)))] * (alphaF[((int)(int(__gid.x)))] - alpha[o]));
      }
    }
    stf[((int)(int(__gid.x)))] = ((paramsF[0] * Kf) * snGrad);
  }
}
