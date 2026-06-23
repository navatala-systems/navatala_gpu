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

__kernel void navatala_sparse_interpolate_distance2_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const uint* strongMask, __global const int* cfMarking, __global const uint* diagIdx, __global const uint* ProwPtr, __global const uint* nRows, __global uint* PcolIdx, __global float* Pvalues) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    int pStart = ((int)(ProwPtr[row]));
    if (mark == 1) {
      PcolIdx[pStart] = ((uint)(row));
      Pvalues[pStart] = as_float(0x3f800000u);
    } else {
      if (mark == -1) {
        int iDiagPos = ((int)(diagIdx[row]));
        float diagI = values[iDiagPos];
        int pos = 0;
        for (int jw = 0; jw < (int)((re - rs)); ++jw) {
          int kw = (rs + jw);
          uint isStrW = strongMask[kw];
          if (isStrW == (uint)(1u)) {
            int colW = ((int)(colIdx[kw]));
            int mW = cfMarking[colW];
            if (mW == 1) {
              float aij = values[kw];
              float w = (-(aij / diagI));
              PcolIdx[(pStart + pos)] = ((uint)(colW));
              Pvalues[(pStart + pos)] = w;
              pos = (pos + 1);
            }
          }
        }
        for (int jf = 0; jf < (int)((re - rs)); ++jf) {
          int kf = (rs + jf);
          uint isStrF = strongMask[kf];
          if (isStrF == (uint)(1u)) {
            int nbrF = ((int)(colIdx[kf]));
            int mF = cfMarking[nbrF];
            if (mF == -1) {
              float aij = values[kf];
              int jDiagPos = ((int)(diagIdx[nbrF]));
              float diagJ = values[jDiagPos];
              int rs2 = ((int)(rowPtr[nbrF]));
              int re2 = ((int)(rowPtr[(nbrF + 1)]));
              for (int j2 = 0; j2 < (int)((re2 - rs2)); ++j2) {
                int k2 = (rs2 + j2);
                uint isStr2 = strongMask[k2];
                if (isStr2 == (uint)(1u)) {
                  int nbr2 = ((int)(colIdx[k2]));
                  int m2 = cfMarking[nbr2];
                  if (m2 == 1) {
                    float ajk = values[k2];
                    float prod = (aij * (ajk / diagJ));
                    float wff = (-(prod / diagI));
                    PcolIdx[(pStart + pos)] = ((uint)(nbr2));
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
