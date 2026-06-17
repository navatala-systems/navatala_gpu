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

__kernel void navatala_sparse_cf_split_aggressive_h_m_i_s(__global const uint* rowPtr, __global const uint* colIdx, __global const uint* strongMask, __global const int* cfMarkingPrev, __global const uint* nRows, __global const uint* seed, __global int* cfMarking) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int prevMark = cfMarkingPrev[row];
    if ((prevMark == 1)) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      uint deg = (uint)(0u);
      for (int jd = 0; jd < (int)((re - rs)); ++jd) {
        int kd = (rs + jd);
        uint isStr = strongMask[kd];
        if ((isStr == (uint)(1u))) {
          int nbrD = ((int)(colIdx[kd]));
          int nbrPD = cfMarkingPrev[nbrD];
          if ((nbrPD == 1)) {
            deg = (deg + (uint)(1u));
          }
        }
      }
      uint priority = ((((uint)(row)) ^ seed[0]) + deg);
      bool hasHigherNbr = false;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStr = strongMask[k];
        if ((isStr == (uint)(1u))) {
          int nbr = ((int)(colIdx[k]));
          int nbrP = cfMarkingPrev[nbr];
          if ((nbrP == 1)) {
            uint nbrRs = rowPtr[nbr];
            uint nbrRe = rowPtr[(nbr + 1)];
            uint nbrDeg = (nbrRe - nbrRs);
            uint nbrPri = ((((uint)(nbr)) ^ seed[0]) + nbrDeg);
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
