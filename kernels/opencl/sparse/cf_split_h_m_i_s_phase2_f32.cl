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

__kernel void navatala_sparse_cf_split_h_m_i_s_phase2_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const uint* strongMask, __global const uint* nRows, __global int* cfMarking) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int mark = cfMarking[row];
    if ((mark == -1)) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      bool hasCoarseNbr = false;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStrong = strongMask[k];
        if ((isStrong == (uint)(1u))) {
          int nbr = ((int)(colIdx[k]));
          int nbrMark = cfMarking[nbr];
          if ((nbrMark == 1)) {
            hasCoarseNbr = true;
          }
        }
      }
      if (hasCoarseNbr) {
      } else {
        cfMarking[row] = 1;
      }
    }
  }
}
