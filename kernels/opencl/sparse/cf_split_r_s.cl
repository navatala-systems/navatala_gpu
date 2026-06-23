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

__kernel void navatala_sparse_cf_split_r_s(__global const uint* rowPtr, __global const uint* colIdx, __global const uint* strongMask, __global const uint* nRows, __global int* cfMarking) {
  int gid0 = (int)get_global_id(0);
  int N = ((int)(nRows[0]));
  if ((int)(get_global_id(0)) == 0) {
    for (int row = 0; row < (int)(N); ++row) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      uint nStrongC = (uint)(0u);
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        uint isStrong = strongMask[k];
        if (isStrong == (uint)(1u)) {
          int nbr = ((int)(colIdx[k]));
          int nbrMark = cfMarking[nbr];
          if (nbrMark == 1) {
            nStrongC = (nStrongC + (uint)(1u));
          }
        }
      }
      if (nStrongC == (uint)(0u)) {
        cfMarking[row] = 1;
      } else {
        cfMarking[row] = -1;
      }
    }
  }
}
