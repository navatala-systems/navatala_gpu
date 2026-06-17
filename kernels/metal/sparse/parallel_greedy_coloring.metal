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

kernel void navatala_sparse_parallel_greedy_coloring(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* nRows [[buffer(2)]], device const uint* seed [[buffer(3)]], device int* colors [[buffer(4)]], device uint* nColors [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if ((gid < N)) {
    int myColor = colors[gid];
    if ((myColor < 0)) {
      int seedI = ((int)(seed[0]));
      int myHash = (((((((gid ^ seedI) + 2127912214) ^ ((gid ^ seedI) << 12)) ^ 3345072700) + ((((gid ^ seedI) + 2127912214) ^ ((gid ^ seedI) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ seedI) + 2127912214) ^ ((gid ^ seedI) << 12)) ^ 3345072700) + ((((gid ^ seedI) + 2127912214) ^ ((gid ^ seedI) << 12)) >> 7)) << 5));
      int rs = ((int)(rowPtr[gid]));
      int re = ((int)(rowPtr[(gid + 1)]));
      int isMax = 1;
      int usedMask = 0;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int col = ((int)(colIdx[k]));
        if ((col != gid)) {
          int cc = colors[col];
          if ((cc >= 0)) {
            usedMask = (usedMask | (1 << cc));
          } else {
            int nHash = (((((((col ^ seedI) + 2127912214) ^ ((col ^ seedI) << 12)) ^ 3345072700) + ((((col ^ seedI) + 2127912214) ^ ((col ^ seedI) << 12)) >> 7)) + 374761393) ^ ((((((col ^ seedI) + 2127912214) ^ ((col ^ seedI) << 12)) ^ 3345072700) + ((((col ^ seedI) + 2127912214) ^ ((col ^ seedI) << 12)) >> 7)) << 5));
            if ((nHash >= myHash)) {
              isMax = 0;
            }
          }
        }
      }
      if ((isMax == 1)) {
        int myC = 0;
        for (int c = 0; c < (int)(32); ++c) {
          if (((usedMask & (1 << myC)) == 0)) {
          } else {
            myC = (myC + 1);
          }
        }
        colors[gid] = myC;
      }
    }
  }
}
