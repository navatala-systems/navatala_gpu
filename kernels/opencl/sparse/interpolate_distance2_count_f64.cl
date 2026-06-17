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

__kernel void navatala_sparse_interpolate_distance2_count_f64(__global const uint* rowPtr, __global const uint* colIdx, __global const uint* strongMask, __global const int* cfMarking, __global const uint* nRows, __global uint* nnzPerRow) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int mark = cfMarking[row];
    if ((mark == 1)) {
      nnzPerRow[row] = (uint)(1u);
    } else {
      if ((mark == -1)) {
        int rs = ((int)(rowPtr[row]));
        int re = ((int)(rowPtr[(row + 1)]));
        uint nnz = (uint)(0u);
        for (int j1 = 0; j1 < (int)((re - rs)); ++j1) {
          int k1 = (rs + j1);
          uint isStr1 = strongMask[k1];
          if ((isStr1 == (uint)(1u))) {
            int nbr1 = ((int)(colIdx[k1]));
            int m1 = cfMarking[nbr1];
            if ((m1 == 1)) {
              nnz = (nnz + (uint)(1u));
            } else {
              if ((m1 == -1)) {
                int rs2 = ((int)(rowPtr[nbr1]));
                int re2 = ((int)(rowPtr[(nbr1 + 1)]));
                for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                  int k2 = (rs2 + j2);
                  uint isStr2 = strongMask[k2];
                  if ((isStr2 == (uint)(1u))) {
                    int nbr2 = ((int)(colIdx[k2]));
                    int m2 = cfMarking[nbr2];
                    if ((m2 == 1)) {
                      nnz = (nnz + (uint)(1u));
                    }
                  }
                }
              }
            }
          }
        }
        nnzPerRow[row] = nnz;
      } else {
        nnzPerRow[row] = (uint)(0u);
      }
    }
  }
}
