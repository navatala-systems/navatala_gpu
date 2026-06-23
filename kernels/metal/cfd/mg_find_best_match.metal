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

kernel void navatala_cfd_mg_find_best_match(device const uint* csrOffsets [[buffer(0)]], device const uint* csrFaceIdx [[buffer(1)]], device const int* csrSign [[buffer(2)]], device const float* strength [[buffer(3)]], device const int* owner [[buffer(4)]], device const int* nei [[buffer(5)]], device int* bestNeighbor [[buffer(6)]], device const int* counts [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float bestS = as_type<float>(0xbf800000u);
    int bestN = -1;
    uint beg = csrOffsets[int(__gid.x)];
    int c1 = (int(__gid.x) + 1);
    uint end = csrOffsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      uint f = csrFaceIdx[k];
      float s = strength[((int)(f))];
      if (s > bestS) {
        bestS = s;
        int sg = csrSign[k];
        if (sg > 0) {
          bestN = nei[((int)(f))];
        } else {
          bestN = owner[((int)(f))];
        }
      }
    }
    bestNeighbor[int(__gid.x)] = bestN;
  }
}
