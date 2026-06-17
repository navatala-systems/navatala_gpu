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
extern "C" __global__ void navatala_sparse_csr_to_ell_f64(const unsigned int* rowPtr, const unsigned int* colIdx, const double* values, const unsigned int* nRows, const unsigned int* maxNnzPerRow, unsigned int* ellColIdx, double* ellValues) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if ((gid < N)) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int nnz = (re - rs);
    int maxNnz = ((int)(maxNnzPerRow[0]));
    for (int j = 0; j < (int)(maxNnz); ++j) {
      if ((j < nnz)) {
        int k = (rs + j);
        ellColIdx[((gid * maxNnz) + j)] = colIdx[k];
        ellValues[((gid * maxNnz) + j)] = values[k];
      } else {
        ellColIdx[((gid * maxNnz) + j)] = 4294967295u;
        ellValues[((gid * maxNnz) + j)] = __longlong_as_double(0x0000000000000000ull);
      }
    }
  }
}
