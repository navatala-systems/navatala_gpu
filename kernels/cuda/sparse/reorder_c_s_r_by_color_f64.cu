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
extern "C" __global__ void navatala_sparse_reorder_c_s_r_by_color_f64(const unsigned int* rowPtr, const unsigned int* colIdx, const double* values, const unsigned int* reorderPerm, const unsigned int* nRows, unsigned int* newRowPtr, unsigned int* newColIdx, double* newValues) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int origRow = ((int)(reorderPerm[gid]));
    int rs = ((int)(rowPtr[origRow]));
    int re = ((int)(rowPtr[(origRow + 1)]));
    int nnz = (re - rs);
    newRowPtr[gid] = ((unsigned int)(nnz));
    for (int j = 0; j < (int)(nnz); ++j) {
      int k = (rs + j);
      newColIdx[(gid + j)] = colIdx[k];
      newValues[(gid + j)] = values[k];
    }
  }
}
