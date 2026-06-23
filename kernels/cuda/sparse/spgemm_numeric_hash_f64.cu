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
extern "C" __global__ void navatala_sparse_spgemm_numeric_hash_f64(const unsigned int* ArowPtr, const unsigned int* AcolIdx, const double* Avalues, const unsigned int* BrowPtr, const unsigned int* BcolIdx, const double* Bvalues, const unsigned int* CrowPtr, const unsigned int* nRowsA, const unsigned int* hashSize, unsigned int* CcolIdx, double* Cvalues) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRowsA[0]));
  if (gid < N) {
    int ars = ((int)(ArowPtr[gid]));
    int are = ((int)(ArowPtr[(gid + 1)]));
    int crs = ((int)(CrowPtr[gid]));
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
        unsigned int bColIdx = BcolIdx[bk];
        double bVal = Bvalues[bk];
        CcolIdx[tgt] = bColIdx;
        Cvalues[tgt] = (aVal * bVal);
        ci = (ci + 1);
      }
    }
  }
}
