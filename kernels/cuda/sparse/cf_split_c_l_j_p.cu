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
extern "C" __global__ void navatala_sparse_cf_split_c_l_j_p(const unsigned int* rowPtr, const unsigned int* colIdx, const unsigned int* strongMask, const float* weights, const unsigned int* nRows, const unsigned int* seed, int* cfMarking) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    unsigned int hash = (((unsigned int)(row)) ^ seed[0]);
    float w = weights[row];
    unsigned int priority = (hash + ((unsigned int)(w)));
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    bool isHighest = true;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      unsigned int isStr = strongMask[k];
      if ((isStr == 1u)) {
        int nbr = ((int)(colIdx[k]));
        unsigned int nbrHash = (((unsigned int)(nbr)) ^ seed[0]);
        float nbrW = weights[nbr];
        unsigned int nbrPri = (nbrHash + ((unsigned int)(nbrW)));
        if ((nbrPri > priority)) {
          isHighest = false;
        }
      }
    }
    if (isHighest) {
      cfMarking[row] = 1;
    } else {
      bool hasCoarse = false;
      for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
        int k2 = (rs + j2);
        unsigned int isStr2 = strongMask[k2];
        if ((isStr2 == 1u)) {
          int nbr2 = ((int)(colIdx[k2]));
          int nbrM = cfMarking[nbr2];
          if ((nbrM == 1)) {
            hasCoarse = true;
          }
        }
      }
      if (hasCoarse) {
        cfMarking[row] = -1;
      }
    }
  }
}
