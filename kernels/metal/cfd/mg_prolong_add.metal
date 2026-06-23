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

kernel void navatala_cfd_mg_prolong_add(device const float* xCoarse [[buffer(0)]], device const int* prolongRowOffsets [[buffer(1)]], device const int* prolongCols [[buffer(2)]], device const float* prolongVals [[buffer(3)]], device float* zFine [[buffer(4)]], device const int* mgCounts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(mgCounts[0]))) {
    return;
  } else {
    float accum = as_type<float>(0x00000000u);
    int beg = prolongRowOffsets[int(__gid.x)];
    int c1 = (int(__gid.x) + 1);
    int end = prolongRowOffsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int col = prolongCols[k];
      accum = (accum + (prolongVals[k] * xCoarse[col]));
    }
    zFine[int(__gid.x)] = (zFine[int(__gid.x)] + accum);
  }
}
