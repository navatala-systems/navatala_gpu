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
extern "C" __global__ void navatala_sparse_csr_sort_columns_f64(const unsigned int* rowPtr, const unsigned int* nRows, unsigned int* colIdx, double* values) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if ((gid < N)) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int len = (re - rs);
    for (int i = 0; i < (int)((len - 1)); ++i) {
      for (int j = 0; j < (int)(((len - 1) - i)); ++j) {
        int a = (rs + j);
        int b = (rs + (j + 1));
        unsigned int ca = colIdx[a];
        unsigned int cb = colIdx[b];
        if ((ca > cb)) {
          colIdx[a] = cb;
          colIdx[b] = ca;
          double va = values[a];
          double vb = values[b];
          values[a] = vb;
          values[b] = va;
        }
      }
    }
  }
}
