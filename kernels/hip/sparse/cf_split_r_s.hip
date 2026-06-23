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
extern "C" __global__ void navatala_sparse_cf_split_r_s(const unsigned int* rowPtr, const unsigned int* colIdx, const unsigned int* strongMask, const unsigned int* nRows, int* cfMarking) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    for (int row = 0; row < (int)(N); ++row) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      unsigned int nStrongC = 0u;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        unsigned int isStrong = strongMask[k];
        if (isStrong == 1u) {
          int nbr = ((int)(colIdx[k]));
          int nbrMark = cfMarking[nbr];
          if (nbrMark == 1) {
            nStrongC = (nStrongC + 1u);
          }
        }
      }
      if (nStrongC == 0u) {
        cfMarking[row] = 1;
      } else {
        cfMarking[row] = -1;
      }
    }
  }
}
