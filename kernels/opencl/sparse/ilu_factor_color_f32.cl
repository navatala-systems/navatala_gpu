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

__kernel void navatala_sparse_ilu_factor_color_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const uint* diagIdx, __global const uint* colorOffsets, __global const uint* nRowsThisColor, __global float* values) {
  int gid0 = (int)get_global_id(0);
  int localIdx = (int)(get_global_id(0));
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int offset = ((int)(colorOffsets[0]));
    int row = (offset + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    for (int jj = 0; jj < (int)((re - rs)); ++jj) {
      int k = (rs + jj);
      int col = ((int)(colIdx[k]));
      if (col < row) {
        int colDiagPos = ((int)(diagIdx[col]));
        float diagVal = values[colDiagPos];
        float lFactor = (values[k] / diagVal);
        values[k] = lFactor;
        int colRs = ((int)(rowPtr[col]));
        int colRe = ((int)(rowPtr[(col + 1)]));
        for (int mm = 0; mm < (int)((re - (k + 1))); ++mm) {
          int m = ((k + 1) + mm);
          int mcol = ((int)(colIdx[m]));
          for (int pp = 0; pp < (int)((colRe - colRs)); ++pp) {
            int p = (colRs + pp);
            int pcol = ((int)(colIdx[p]));
            if (pcol == mcol) {
              float uVal = values[p];
              float aim = values[m];
              values[m] = (aim - (lFactor * uVal));
            }
          }
        }
      }
    }
  }
}
