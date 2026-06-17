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

__kernel void navatala_sparse_cf_split_p_m_i_s(__global const uint* rowPtr, __global const uint* colIdx, __global const uint* strongMask, __global const uint* nRows, __global const uint* seed, __global int* cfMarking) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    uint priority = (((uint)(row)) ^ seed[0]);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    bool hasHigherNbr = false;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      uint isStrong = strongMask[k];
      if ((isStrong == (uint)(1u))) {
        int nbr = ((int)(colIdx[k]));
        int nbrMark = cfMarking[nbr];
        if ((nbrMark == 0)) {
          uint nbrPri = (((uint)(nbr)) ^ seed[0]);
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
      bool hasCoarseNbr = false;
      for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
        int k2 = (rs + j2);
        uint isStrong2 = strongMask[k2];
        if ((isStrong2 == (uint)(1u))) {
          int nbr2 = ((int)(colIdx[k2]));
          int nbrMark2 = cfMarking[nbr2];
          if ((nbrMark2 == 1)) {
            hasCoarseNbr = true;
          }
        }
      }
      if (hasCoarseNbr) {
        cfMarking[row] = -1;
      }
    }
  }
}
