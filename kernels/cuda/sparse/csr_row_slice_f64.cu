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
extern "C" __global__ void navatala_sparse_csr_row_slice_f64(const unsigned int* rowPtr, const unsigned int* colIdx, const double* values, const unsigned int* rowBegin, const unsigned int* rowEnd, unsigned int* subRowPtr, unsigned int* subColIdx, double* subValues) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int begin = ((int)(rowBegin[0]));
  int end = ((int)(rowEnd[0]));
  int nRows = (end - begin);
  if ((gid < nRows)) {
    int origRow = (begin + gid);
    int rs = ((int)(rowPtr[origRow]));
    int re = ((int)(rowPtr[(origRow + 1)]));
    int subRs = ((int)(rowPtr[begin]));
    subRowPtr[gid] = ((unsigned int)((rs - subRs)));
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      subColIdx[k] = colIdx[k];
      subValues[k] = values[k];
    }
  }
}
