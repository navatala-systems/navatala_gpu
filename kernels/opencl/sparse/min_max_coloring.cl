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

__kernel void navatala_sparse_min_max_coloring(__global const uint* rowPtr, __global const uint* colIdx, __global const uint* nRows, __global int* colors, __global uint* nColors) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
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
