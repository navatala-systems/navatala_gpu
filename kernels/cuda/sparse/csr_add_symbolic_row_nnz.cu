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
extern "C" __global__ void navatala_sparse_csr_add_symbolic_row_nnz(const unsigned int* ArowPtr, const unsigned int* AcolIdx, const unsigned int* BrowPtr, const unsigned int* BcolIdx, const unsigned int* nRows, unsigned int* CrowNnz) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int ars = ((int)(ArowPtr[gid]));
    int are = ((int)(ArowPtr[(gid + 1)]));
    int brs = ((int)(BrowPtr[gid]));
    int bre = ((int)(BrowPtr[(gid + 1)]));
    int annz = (are - ars);
    int bnnz = (bre - brs);
    int count = 0;
    int ai = 0;
    int bi = 0;
    for (int __iter = 0; __iter < 256; ++__iter) {
      if (!((ai < annz) && (bi < bnnz))) break;
      int acol = ((int)(AcolIdx[(ars + ai)]));
      int bcol = ((int)(BcolIdx[(brs + bi)]));
      if (acol == bcol) {
        count = (count + 1);
        ai = (ai + 1);
        bi = (bi + 1);
      } else {
        if (acol < bcol) {
          count = (count + 1);
          ai = (ai + 1);
        } else {
          count = (count + 1);
          bi = (bi + 1);
        }
      }
    }
    count = (count + (annz - ai));
    count = (count + (bnnz - bi));
    CrowNnz[gid] = ((unsigned int)(count));
  }
}
