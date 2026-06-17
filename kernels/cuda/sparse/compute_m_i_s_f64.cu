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
extern "C" __global__ void navatala_sparse_compute_m_i_s_f64(const unsigned int* strongRowPtr, const unsigned int* strongColIdx, const unsigned int* randomPriority, const unsigned int* nCells, int* aggregateId, unsigned int* nUndecided) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int cell = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nCells[0]));
  if ((cell < N)) {
    int myAgg = aggregateId[cell];
    if ((myAgg == -1)) {
      unsigned int myPri = randomPriority[cell];
      int rs = ((int)(strongRowPtr[cell]));
      int re = ((int)(strongRowPtr[(cell + 1)]));
      int isMax = 1;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int nbr = ((int)(strongColIdx[k]));
        int nbrAgg = aggregateId[nbr];
        if ((nbrAgg == -1)) {
          unsigned int nbrPri = randomPriority[nbr];
          if (((nbrPri > myPri) || ((nbrPri == myPri) && (nbr < cell)))) {
            isMax = 0;
          }
        }
      }
      if ((isMax == 1)) {
        aggregateId[cell] = cell;
      } else {
        int assigned = 0;
        for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
          if ((assigned == 0)) {
            int k2 = (rs + j2);
            int nbr2 = ((int)(strongColIdx[k2]));
            int nbrAgg2 = aggregateId[nbr2];
            if ((nbrAgg2 >= 0)) {
              aggregateId[cell] = nbrAgg2;
              assigned = 1;
            }
          }
        }
        if ((assigned == 0)) {
          unsigned int _u = atomicAdd(&(nUndecided[0]), 1u);
        }
      }
    }
  }
}
