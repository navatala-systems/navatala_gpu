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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_dilu_build_reciprocal_d_f64(const unsigned int* rowPtr, const unsigned int* colIdx, const double* values, const unsigned int* colorOffsets, const unsigned int* nRowsThisColor, double* recipD) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int localIdx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int offset = ((int)(colorOffsets[0]));
    int row = (offset + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    double diag = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col == row) {
        diag = values[k];
      }
    }
    double sum = __longlong_as_double(0x0000000000000000ull);
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      int col2 = ((int)(colIdx[k2]));
      if (col2 < row) {
        double aij = values[k2];
        double rdj = recipD[col2];
        int jrs = ((int)(rowPtr[col2]));
        int jre = ((int)(rowPtr[(col2 + 1)]));
        double aji = __longlong_as_double(0x0000000000000000ull);
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
    recipD[row] = (__longlong_as_double(0x3ff0000000000000ull) / (diag - sum));
  }
}
