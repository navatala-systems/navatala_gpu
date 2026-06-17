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

kernel void navatala_sparse_interpolate_distance2_count_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const uint* strongMask [[buffer(2)]], device const int* cfMarking [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* nnzPerRow [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int mark = cfMarking[row];
    if ((mark == 1)) {
      nnzPerRow[row] = 1u;
    } else {
      if ((mark == -1)) {
        int rs = ((int)(rowPtr[row]));
        int re = ((int)(rowPtr[(row + 1)]));
        uint nnz = 0u;
        for (int j1 = 0; j1 < (int)((re - rs)); ++j1) {
          int k1 = (rs + j1);
          uint isStr1 = strongMask[k1];
          if ((isStr1 == 1u)) {
            int nbr1 = ((int)(colIdx[k1]));
            int m1 = cfMarking[nbr1];
            if ((m1 == 1)) {
              nnz = (nnz + 1u);
            } else {
              if ((m1 == -1)) {
                int rs2 = ((int)(rowPtr[nbr1]));
                int re2 = ((int)(rowPtr[(nbr1 + 1)]));
                for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                  int k2 = (rs2 + j2);
                  uint isStr2 = strongMask[k2];
                  if ((isStr2 == 1u)) {
                    int nbr2 = ((int)(colIdx[k2]));
                    int m2 = cfMarking[nbr2];
                    if ((m2 == 1)) {
                      nnz = (nnz + 1u);
                    }
                  }
                }
              }
            }
          }
        }
        nnzPerRow[row] = nnz;
      } else {
        nnzPerRow[row] = 0u;
      }
    }
  }
}
