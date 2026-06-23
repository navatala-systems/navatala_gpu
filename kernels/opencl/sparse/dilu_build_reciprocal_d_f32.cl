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

__kernel void navatala_sparse_dilu_build_reciprocal_d_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const uint* colorOffsets, __global const uint* nRowsThisColor, __global float* recipD) {
  int gid0 = (int)get_global_id(0);
  int localIdx = (int)(get_global_id(0));
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int offset = ((int)(colorOffsets[0]));
    int row = (offset + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float diag = as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col == row) {
        diag = values[k];
      }
    }
    float sum = as_float(0x00000000u);
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      int col2 = ((int)(colIdx[k2]));
      if (col2 < row) {
        float aij = values[k2];
        float rdj = recipD[col2];
        int jrs = ((int)(rowPtr[col2]));
        int jre = ((int)(rowPtr[(col2 + 1)]));
        float aji = as_float(0x00000000u);
        for (int pp = 0; pp < (int)((jre - jrs)); ++pp) {
          int p = (jrs + pp);
          int pcol = ((int)(colIdx[p]));
          if (pcol == row) {
            aji = values[p];
          }
        }
        sum = (sum + (aij * (aji * rdj)));
      }
    }
    recipD[row] = (as_float(0x3f800000u) / (diag - sum));
  }
}
