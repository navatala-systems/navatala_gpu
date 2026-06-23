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

__kernel void navatala_sparse_interpolate_f_f_weights_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const uint* strongMask, __global const int* cfMarking, __global const float* diag, __global const uint* ProwPtr, __global const uint* PcolIdx, __global const uint* nRows, __global float* Pvalues) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
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
        uint isStr = strongMask[k];
        if (isStr == (uint)(1u)) {
          int col = ((int)(colIdx[k]));
          int mC = cfMarking[col];
          if (mC == 1) {
            float aij = values[k];
            float numer = aij;
            for (int ff = 0; ff < (int)((re - rs)); ++ff) {
              int kf = (rs + ff);
              uint isStrF = strongMask[kf];
              if (isStrF == (uint)(1u)) {
                int colF = ((int)(colIdx[kf]));
                int mF = cfMarking[colF];
                if (mF == -1) {
                  float aik = values[kf];
                  float diagK = diag[colF];
                  int rsK = ((int)(rowPtr[colF]));
                  int reK = ((int)(rowPtr[(colF + 1)]));
                  float akj = as_float(0x00000000u);
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
