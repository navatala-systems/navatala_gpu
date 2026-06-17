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
extern "C" __global__ void navatala_sparse_cf_split_aggressive_p_m_i_s(const unsigned int* rowPtr, const unsigned int* colIdx, const unsigned int* strongMask, const int* cfMarkingPrev, const unsigned int* nRows, const unsigned int* seed, int* cfMarking) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int prevMark = cfMarkingPrev[row];
    if ((prevMark == 1)) {
      unsigned int priority = (((unsigned int)(row)) ^ seed[0]);
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      bool hasHigherNbr = false;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        unsigned int isStr = strongMask[k];
        if ((isStr == 1u)) {
          int nbr = ((int)(colIdx[k]));
          int nbrPrev = cfMarkingPrev[nbr];
          if ((nbrPrev == 1)) {
            unsigned int nbrPri = (((unsigned int)(nbr)) ^ seed[0]);
            bool outranks = ((nbrPri > priority) || ((nbrPri == priority) && (nbr < row)));
            if (outranks) {
              hasHigherNbr = true;
            }
          }
        }
      }
      if ((!hasHigherNbr)) {
        cfMarking[row] = 1;
      } else {
        cfMarking[row] = -1;
      }
    } else {
      cfMarking[row] = prevMark;
    }
  }
}
