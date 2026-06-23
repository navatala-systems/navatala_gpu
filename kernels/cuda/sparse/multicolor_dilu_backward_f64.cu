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
extern "C" __global__ void navatala_sparse_multicolor_dilu_backward_f64(const unsigned int* rowPtr, const unsigned int* colIdx, const double* values, const double* recipD, const double* b, const unsigned int* colorOffsets, const unsigned int* nRowsThisColor, double* x) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int localIdx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int offset = ((int)(colorOffsets[0]));
    int row = (offset + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    double sum = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col > row) {
        double a = values[k];
        double xj = x[col];
        sum = (sum + (a * xj));
      }
    }
    double rd = recipD[row];
    double bi = b[row];
    x[row] = (rd * (bi - sum));
  }
}
