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
extern "C" __global__ void navatala_sparse_interpolate_distance1_f32(const unsigned int* rowPtr, const unsigned int* colIdx, const float* values, const unsigned int* strongMask, const int* cfMarking, const unsigned int* ProwPtr, const unsigned int* nRows, unsigned int* PcolIdx, float* Pvalues) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    int mark = cfMarking[row];
    int pStart = ((int)(ProwPtr[row]));
    if (mark == 1) {
      PcolIdx[pStart] = ((unsigned int)(row));
      Pvalues[pStart] = __uint_as_float(0x3f800000u);
    } else {
      if (mark == -1) {
        float diag = __uint_as_float(0x00000000u);
        for (int jd = 0; jd < (int)((re - rs)); ++jd) {
          int kd = (rs + jd);
          int colD = ((int)(colIdx[kd]));
          if (colD == row) {
            diag = values[kd];
          }
        }
        int pos = 0;
        for (int jw = 0; jw < (int)((re - rs)); ++jw) {
          int kw = (rs + jw);
          unsigned int isStrW = strongMask[kw];
          if (isStrW == 1u) {
            int colW = ((int)(colIdx[kw]));
            int markW = cfMarking[colW];
            if (markW == 1) {
              float aij = values[kw];
              float w = (-(aij / diag));
              PcolIdx[(pStart + pos)] = ((unsigned int)(colW));
              Pvalues[(pStart + pos)] = w;
              pos = (pos + 1);
            }
          }
        }
      }
    }
  }
}
