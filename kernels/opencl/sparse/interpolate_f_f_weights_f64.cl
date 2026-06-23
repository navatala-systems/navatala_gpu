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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_interpolate_f_f_weights_f64(__global const uint* rowPtr, __global const uint* colIdx, __global const double* values, __global const uint* strongMask, __global const int* cfMarking, __global const double* diag, __global const uint* ProwPtr, __global const uint* PcolIdx, __global const uint* nRows, __global double* Pvalues) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == -1) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      double diagVal = diag[row];
      int pStart = ((int)(ProwPtr[row]));
      int pos = 0;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStr = strongMask[k];
        if (isStr == (uint)(1u)) {
          int col = ((int)(colIdx[k]));
          int mC = cfMarking[col];
          if (mC == 1) {
            double aij = values[k];
            double numer = aij;
            for (int ff = 0; ff < (int)((re - rs)); ++ff) {
              int kf = (rs + ff);
              uint isStrF = strongMask[kf];
              if (isStrF == (uint)(1u)) {
                int colF = ((int)(colIdx[kf]));
                int mF = cfMarking[colF];
                if (mF == -1) {
                  double aik = values[kf];
                  double diagK = diag[colF];
                  int rsK = ((int)(rowPtr[colF]));
                  int reK = ((int)(rowPtr[(colF + 1)]));
                  double akj = as_double(0x0000000000000000ul);
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
            double w = (-(numer / diagVal));
            Pvalues[(pStart + pos)] = w;
            pos = (pos + 1);
          }
        }
      }
    }
  }
}
