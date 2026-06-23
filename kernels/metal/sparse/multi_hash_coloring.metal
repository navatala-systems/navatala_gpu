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

kernel void navatala_sparse_multi_hash_coloring(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* nRows [[buffer(2)]], device const uint* nHashFunctions [[buffer(3)]], device int* colors [[buffer(4)]], device uint* nColors [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int myColor = colors[gid];
    if (myColor < 0) {
      int K = ((int)(nHashFunctions[0]));
      int h0 = (((((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) ^ 3345072700) + ((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) ^ 3345072700) + ((((gid ^ 0) + 2127912214) ^ ((gid ^ 0) << 12)) >> 7)) << 5));
      int h1 = (((((((gid ^ 1043) + 2127912214) ^ ((gid ^ 1043) << 12)) ^ 3345072700) + ((((gid ^ 1043) + 2127912214) ^ ((gid ^ 1043) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ 1043) + 2127912214) ^ ((gid ^ 1043) << 12)) ^ 3345072700) + ((((gid ^ 1043) + 2127912214) ^ ((gid ^ 1043) << 12)) >> 7)) << 5));
      int h2 = (((((((gid ^ 2086) + 2127912214) ^ ((gid ^ 2086) << 12)) ^ 3345072700) + ((((gid ^ 2086) + 2127912214) ^ ((gid ^ 2086) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ 2086) + 2127912214) ^ ((gid ^ 2086) << 12)) ^ 3345072700) + ((((gid ^ 2086) + 2127912214) ^ ((gid ^ 2086) << 12)) >> 7)) << 5));
      int h3 = (((((((gid ^ 3129) + 2127912214) ^ ((gid ^ 3129) << 12)) ^ 3345072700) + ((((gid ^ 3129) + 2127912214) ^ ((gid ^ 3129) << 12)) >> 7)) + 374761393) ^ ((((((gid ^ 3129) + 2127912214) ^ ((gid ^ 3129) << 12)) ^ 3345072700) + ((((gid ^ 3129) + 2127912214) ^ ((gid ^ 3129) << 12)) >> 7)) << 5));
      int notMin = 0;
      int notMax = 0;
      int rs = ((int)(rowPtr[gid]));
      int re = ((int)(rowPtr[(gid + 1)]));
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int col = ((int)(colIdx[k]));
        if ((col != gid) && (col < N)) {
          int cc = colors[col];
          if (cc < 0) {
            int j0 = (((((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) ^ 3345072700) + ((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) >> 7)) + 374761393) ^ ((((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) ^ 3345072700) + ((((col ^ 0) + 2127912214) ^ ((col ^ 0) << 12)) >> 7)) << 5));
            if (j0 >= h0) {
              notMax = (notMax | 1);
            }
            if (j0 <= h0) {
              notMin = (notMin | 1);
            }
            if (K > 1) {
              int j1 = (((((((col ^ 1043) + 2127912214) ^ ((col ^ 1043) << 12)) ^ 3345072700) + ((((col ^ 1043) + 2127912214) ^ ((col ^ 1043) << 12)) >> 7)) + 374761393) ^ ((((((col ^ 1043) + 2127912214) ^ ((col ^ 1043) << 12)) ^ 3345072700) + ((((col ^ 1043) + 2127912214) ^ ((col ^ 1043) << 12)) >> 7)) << 5));
              if (j1 >= h1) {
                notMax = (notMax | 2);
              }
              if (j1 <= h1) {
                notMin = (notMin | 2);
              }
            }
            if (K > 2) {
              int j2 = (((((((col ^ 2086) + 2127912214) ^ ((col ^ 2086) << 12)) ^ 3345072700) + ((((col ^ 2086) + 2127912214) ^ ((col ^ 2086) << 12)) >> 7)) + 374761393) ^ ((((((col ^ 2086) + 2127912214) ^ ((col ^ 2086) << 12)) ^ 3345072700) + ((((col ^ 2086) + 2127912214) ^ ((col ^ 2086) << 12)) >> 7)) << 5));
              if (j2 >= h2) {
                notMax = (notMax | 4);
              }
              if (j2 <= h2) {
                notMin = (notMin | 4);
              }
            }
            if (K > 3) {
              int j3 = (((((((col ^ 3129) + 2127912214) ^ ((col ^ 3129) << 12)) ^ 3345072700) + ((((col ^ 3129) + 2127912214) ^ ((col ^ 3129) << 12)) >> 7)) + 374761393) ^ ((((((col ^ 3129) + 2127912214) ^ ((col ^ 3129) << 12)) ^ 3345072700) + ((((col ^ 3129) + 2127912214) ^ ((col ^ 3129) << 12)) >> 7)) << 5));
              if (j3 >= h3) {
                notMax = (notMax | 8);
              }
              if (j3 <= h3) {
                notMin = (notMin | 8);
              }
            }
          }
        }
      }
      int picked = -1;
      if ((notMin & 1) == 0) {
        picked = 0;
      }
      if ((picked == -1) && ((notMax & 1) == 0)) {
        picked = 1;
      }
      if ((picked == -1) && (K > 1)) {
        if ((notMin & 2) == 0) {
          picked = 2;
        }
        if ((picked == -1) && ((notMax & 2) == 0)) {
          picked = 3;
        }
      }
      if ((picked == -1) && (K > 2)) {
        if ((notMin & 4) == 0) {
          picked = 4;
        }
        if ((picked == -1) && ((notMax & 4) == 0)) {
          picked = 5;
        }
      }
      if ((picked == -1) && (K > 3)) {
        if ((notMin & 8) == 0) {
          picked = 6;
        }
        if ((picked == -1) && ((notMax & 8) == 0)) {
          picked = 7;
        }
      }
      if (picked >= 0) {
        colors[gid] = picked;
      }
    }
  }
}
