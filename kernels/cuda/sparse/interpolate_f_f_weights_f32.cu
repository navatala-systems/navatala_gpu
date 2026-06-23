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
extern "C" __global__ void navatala_sparse_interpolate_f_f_weights_f32(const unsigned int* rowPtr, const unsigned int* colIdx, const float* values, const unsigned int* strongMask, const int* cfMarking, const float* diag, const unsigned int* ProwPtr, const unsigned int* PcolIdx, const unsigned int* nRows, float* Pvalues) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == -1) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      float diagVal = diag[row];
      int pStart = ((int)(ProwPtr[row]));
      int pos = 0;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        unsigned int isStr = strongMask[k];
        if (isStr == 1u) {
          int col = ((int)(colIdx[k]));
          int mC = cfMarking[col];
          if (mC == 1) {
            float aij = values[k];
            float numer = aij;
            for (int ff = 0; ff < (int)((re - rs)); ++ff) {
              int kf = (rs + ff);
              unsigned int isStrF = strongMask[kf];
              if (isStrF == 1u) {
                int colF = ((int)(colIdx[kf]));
                int mF = cfMarking[colF];
                if (mF == -1) {
                  float aik = values[kf];
                  float diagK = diag[colF];
                  int rsK = ((int)(rowPtr[colF]));
                  int reK = ((int)(rowPtr[(colF + 1)]));
                  float akj = __uint_as_float(0x00000000u);
                  for (int kn = 0; kn < (int)((reK - rsK)); ++kn) {
                    int knIdx = (rsK + kn);
                    int knCol = ((int)(colIdx[knIdx]));
                    if (knCol == col) {
                      akj = values[knIdx];
                    }
                  }
                  numer = (numer + (aik * (akj / diagK)));
                }
              }
            }
            float w = (-(numer / diagVal));
            Pvalues[(pStart + pos)] = w;
            pos = (pos + 1);
          }
        }
      }
    }
  }
}
