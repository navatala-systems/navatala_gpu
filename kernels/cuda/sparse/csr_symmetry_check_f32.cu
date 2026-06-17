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
extern "C" __global__ void navatala_sparse_csr_symmetry_check_f32(const unsigned int* rowPtr, const unsigned int* colIdx, const float* values, const float* tol, const unsigned int* nRows, unsigned int* asymmetricFlags, unsigned int* isSymmetric) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if ((gid < N)) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if ((col < gid)) {
      } else {
        if ((col > gid)) {
          float a = values[k];
          int colRs = ((int)(rowPtr[col]));
          int colRe = ((int)(rowPtr[(col + 1)]));
          unsigned int found = 0u;
          for (int jj = 0; jj < (int)((colRe - colRs)); ++jj) {
            int kk = (colRs + jj);
            int cc = ((int)(colIdx[kk]));
            if ((cc == gid)) {
              float at = values[kk];
              float tolVal = tol[0];
              float diff = abs((a - at));
              if ((diff > tolVal)) {
                asymmetricFlags[gid] = 1u;
                unsigned int _aod5 = atomicAnd(&(isSymmetric[0]), 0u);
              }
            }
          }
        }
      }
    }
  }
}
