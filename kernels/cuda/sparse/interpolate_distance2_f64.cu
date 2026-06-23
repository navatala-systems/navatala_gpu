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
extern "C" __global__ void navatala_sparse_interpolate_distance2_f64(const unsigned int* rowPtr, const unsigned int* colIdx, const double* values, const unsigned int* strongMask, const int* cfMarking, const unsigned int* diagIdx, const unsigned int* ProwPtr, const unsigned int* nRows, unsigned int* PcolIdx, double* Pvalues) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    int pStart = ((int)(ProwPtr[row]));
    if (mark == 1) {
      PcolIdx[pStart] = ((unsigned int)(row));
      Pvalues[pStart] = __longlong_as_double(0x3ff0000000000000ull);
    } else {
      if (mark == -1) {
        int iDiagPos = ((int)(diagIdx[row]));
        double diagI = values[iDiagPos];
        int pos = 0;
        for (int jw = 0; jw < (int)((re - rs)); ++jw) {
          int kw = (rs + jw);
          unsigned int isStrW = strongMask[kw];
          if (isStrW == 1u) {
            int colW = ((int)(colIdx[kw]));
            int mW = cfMarking[colW];
            if (mW == 1) {
              double aij = values[kw];
              double w = (-(aij / diagI));
              PcolIdx[(pStart + pos)] = ((unsigned int)(colW));
              Pvalues[(pStart + pos)] = w;
              pos = (pos + 1);
            }
          }
        }
        for (int jf = 0; jf < (int)((re - rs)); ++jf) {
          int kf = (rs + jf);
          unsigned int isStrF = strongMask[kf];
          if (isStrF == 1u) {
            int nbrF = ((int)(colIdx[kf]));
            int mF = cfMarking[nbrF];
            if (mF == -1) {
              double aij = values[kf];
              int jDiagPos = ((int)(diagIdx[nbrF]));
              double diagJ = values[jDiagPos];
              int rs2 = ((int)(rowPtr[nbrF]));
              int re2 = ((int)(rowPtr[(nbrF + 1)]));
              for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                int k2 = (rs2 + j2);
                unsigned int isStr2 = strongMask[k2];
                if (isStr2 == 1u) {
                  int nbr2 = ((int)(colIdx[k2]));
                  int m2 = cfMarking[nbr2];
                  if (m2 == 1) {
                    double ajk = values[k2];
                    double prod = (aij * (ajk / diagJ));
                    double wff = (-(prod / diagI));
                    PcolIdx[(pStart + pos)] = ((unsigned int)(nbr2));
                    Pvalues[(pStart + pos)] = wff;
                    pos = (pos + 1);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
