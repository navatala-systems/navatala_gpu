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
extern "C" __global__ void navatala_sparse_csr_sort_after_ldu_fill_f32(const unsigned int* rowPtr, const unsigned int* nRows, unsigned int* colInd, float* values) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int start = ((int)(rowPtr[row]));
    int end = ((int)(rowPtr[(row + 1)]));
    int count = (end - start);
    for (int i = 0; i < (int)(count); ++i) {
      int pos = (start + i);
      unsigned int keyCol = colInd[pos];
      float keyVal = values[pos];
      int j = i;
      for (int __iter = 0; __iter < 64; ++__iter) {
        if (!(((j > 0) && (colInd[((start + j) - 1)] > keyCol)))) break;
        colInd[(start + j)] = colInd[((start + j) - 1)];
        values[(start + j)] = values[((start + j) - 1)];
        j = (j - 1);
      }
      colInd[(start + j)] = keyCol;
      values[(start + j)] = keyVal;
    }
  }
}
