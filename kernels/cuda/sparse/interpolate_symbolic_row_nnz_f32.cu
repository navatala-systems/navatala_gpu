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
extern "C" __global__ void navatala_sparse_interpolate_symbolic_row_nnz_f32(const unsigned int* rowPtr, const unsigned int* colIdx, const unsigned int* strongMask, const int* cfMarking, const unsigned int* nRows, unsigned int* ProwNnz) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int mark = cfMarking[row];
    if ((mark == 1)) {
      ProwNnz[row] = 1u;
    } else {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      unsigned int nnz = 0u;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        unsigned int isStr = strongMask[k];
        if ((isStr == 1u)) {
          int col = ((int)(colIdx[k]));
          int mC = cfMarking[col];
          if ((mC == 1)) {
            nnz = (nnz + 1u);
          }
        }
      }
      ProwNnz[row] = nnz;
    }
  }
}
