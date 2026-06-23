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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_sparse_ilu_factor_color_f64(const unsigned int* rowPtr, const unsigned int* colIdx, const unsigned int* diagIdx, const unsigned int* colorOffsets, const unsigned int* nRowsThisColor, double* values) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int localIdx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
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
        double diagVal = values[colDiagPos];
        double lFactor = (values[k] / diagVal);
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
              double uVal = values[p];
              double aim = values[m];
              values[m] = (aim - (lFactor * uVal));
            }
          }
        }
      }
    }
  }
}
