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

kernel void navatala_sparse_min_max2_ring_coloring(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* nRows [[buffer(2)]], device int* colors [[buffer(3)]], device uint* nColors [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int myColor = colors[gid];
    if (myColor < 0) {
      int myHash = (((((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) ^ 3345072700) + ((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) ^ 3345072700) + ((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) >> 7)) << 5));
      int rs = ((int)(rowPtr[gid]));
      int re = ((int)(rowPtr[(gid + 1)]));
      int isMax = 1;
      int isMin = 1;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int col = ((int)(colIdx[k]));
        if ((col != gid) && (col < N)) {
          int cc = colors[col];
          if (cc < 0) {
            int nHash = (((((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) ^ 3345072700) + ((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) >> 7)) + 374761393) ^ ((((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) ^ 3345072700) + ((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) >> 7)) << 5));
            if (nHash >= myHash) {
              isMax = 0;
            }
            if (nHash <= myHash) {
              isMin = 0;
            }
          }
          int rs2 = ((int)(rowPtr[col]));
          int re2 = ((int)(rowPtr[(col + 1)]));
          for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
            int k2 = (rs2 + j2);
            int col2 = ((int)(colIdx[k2]));
            if ((col2 != gid) && (col2 < N)) {
              int cc2 = colors[col2];
              if (cc2 < 0) {
                int n2Hash = (((((((col2 ^ 0) + 2127912214) ^ ((col2 ^ 0) << 12)) ^ 3345072700) + ((((col2 ^ 0) + 2127912214) ^ ((col2 ^ 0) << 12)) >> 7)) + 374761393) ^ ((((((col2 ^ 0) + 2127912214) ^ ((col2 ^ 0) << 12)) ^ 3345072700) + ((((col2 ^ 0) + 2127912214) ^ ((col2 ^ 0) << 12)) >> 7)) << 5));
                if (n2Hash >= myHash) {
                  isMax = 0;
                }
                if (n2Hash <= myHash) {
                  isMin = 0;
                }
              }
            }
          }
        }
      }
      int round = ((int)(nColors[0]));
      if (isMax == 1) {
        colors[gid] = (2 * round);
      } else {
        if (isMin == 1) {
          colors[gid] = ((2 * round) + 1);
        }
      }
    }
  }
}
