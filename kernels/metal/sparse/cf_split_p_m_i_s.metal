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

kernel void navatala_sparse_cf_split_p_m_i_s(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const uint* nRows [[buffer(3)]], device const uint* seed [[buffer(4)]], device int* cfMarking [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    uint priority = (((uint)(row)) ^ seed[0]);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    bool hasHigherNbr = false;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      uint isStrong = strongMask[k];
      if (isStrong == 1u) {
        int nbr = ((int)(colIdx[k]));
        int nbrMark = cfMarking[nbr];
        if (nbrMark == 0) {
          uint nbrPri = (((uint)(nbr)) ^ seed[0]);
          bool outranks = ((nbrPri > priority) || ((nbrPri == priority) && (nbr < row)));
          if (outranks) {
            hasHigherNbr = true;
          }
        }
      }
    }
    if (!hasHigherNbr) {
      cfMarking[row] = 1;
    } else {
      bool hasCoarseNbr = false;
      for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
        int k2 = (rs + j2);
        uint isStrong2 = strongMask[k2];
        if (isStrong2 == 1u) {
          int nbr2 = ((int)(colIdx[k2]));
          int nbrMark2 = cfMarking[nbr2];
          if (nbrMark2 == 1) {
            hasCoarseNbr = true;
          }
        }
      }
      if (hasCoarseNbr) {
        cfMarking[row] = -1;
      }
    }
  }
}
