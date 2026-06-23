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

kernel void navatala_cfd_sum_mag_off_diag(device const float* upper [[buffer(0)]], device const float* lower [[buffer(1)]], device const int* offsets [[buffer(2)]], device const int* faceIdx [[buffer(3)]], device const int* sign [[buffer(4)]], device const int* counts [[buffer(5)]], device float* outSumOff [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float sum = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < counts[1]) {
        int sgn = sign[k];
        if (sgn > 0) {
          float val = upper[f];
          float negVal = (as_type<float>(0x00000000u) - val);
          float absVal = (((val >= as_type<float>(0x00000000u))) ? (val) : (negVal));
          sum = (sum + absVal);
        } else {
          float val = lower[f];
          float negVal = (as_type<float>(0x00000000u) - val);
          float absVal = (((val >= as_type<float>(0x00000000u))) ? (val) : (negVal));
          sum = (sum + absVal);
        }
      }
    }
    outSumOff[((int)(int(__gid.x)))] = sum;
  }
}
