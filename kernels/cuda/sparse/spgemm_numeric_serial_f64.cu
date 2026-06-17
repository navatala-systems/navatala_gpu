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
extern "C" __global__ void navatala_sparse_spgemm_numeric_serial_f64(const unsigned int* ArowPtr, const unsigned int* AcolIdx, const double* Avalues, const unsigned int* BrowPtr, const unsigned int* BcolIdx, const double* Bvalues, const unsigned int* CrowPtr, const unsigned int* nRowsA, unsigned int* CcolIdx, double* Cvalues) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRowsA[0]));
  for (int row = 0; row < (int)(N); ++row) {
    int ars = ((int)(ArowPtr[row]));
    int are = ((int)(ArowPtr[(row + 1)]));
    int crs = ((int)(CrowPtr[row]));
    int ci = 0;
    for (int j = 0; j < (int)((are - ars)); ++j) {
      int ak = (ars + j);
      double aVal = Avalues[ak];
      int bCol = ((int)(AcolIdx[ak]));
      int brs = ((int)(BrowPtr[bCol]));
      int bre = ((int)(BrowPtr[(bCol + 1)]));
      for (int jj = 0; jj < (int)((bre - brs)); ++jj) {
        int bk = (brs + jj);
        int tgt = (crs + ci);
        CcolIdx[tgt] = BcolIdx[bk];
        Cvalues[tgt] = (aVal * Bvalues[bk]);
        ci = (ci + 1);
      }
    }
  }
}
