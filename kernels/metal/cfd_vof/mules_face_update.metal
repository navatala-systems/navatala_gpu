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

kernel void navatala_cfd_vof_mules_face_update(device const float* phiCorr [[buffer(0)]], device float* lambda [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* nei [[buffer(3)]], device const float* lambdam [[buffer(4)]], device const float* lambdap [[buffer(5)]], device const int* counts [[buffer(6)]], device const float* paramsF [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[2]) {
    return;
  } else {
    float pc = phiCorr[((int)(int(__gid.x)))];
    int o = owner[((int)(int(__gid.x)))];
    int n = nei[((int)(int(__gid.x)))];
    float a0 = lambdam[o];
    float b0 = lambdap[n];
    float lim = a0;
    if (lim > b0) {
      lim = b0;
    }
    if (pc > as_type<float>(0x00000000u)) {
      float a1 = lambdap[o];
      float b1 = lambdam[n];
      float lim1 = a1;
      if (lim1 > b1) {
        lim1 = b1;
      }
      lim = lim1;
    }
    float cur = lambda[((int)(int(__gid.x)))];
    float _out = cur;
    if (_out > lim) {
      _out = lim;
    }
    lambda[((int)(int(__gid.x)))] = _out;
  }
}
