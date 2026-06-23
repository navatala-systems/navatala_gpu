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

__kernel void navatala_sparse_cf_split_c_r_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const uint* strongMask, __global const uint* nRows, __global const float* crThreshold, __global int* cfMarking) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (row < N) {
    int mark = cfMarking[row];
    if (mark == 0) {
      int rs = ((int)(rowPtr[row]));
      int re = ((int)(rowPtr[(row + 1)]));
      float diag = as_float(0x00000000u);
      float offDiagSum = as_float(0x00000000u);
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int col = ((int)(colIdx[k]));
        float a = values[k];
        if (col == row) {
          diag = a;
        } else {
          offDiagSum = (offDiagSum + a);
        }
      }
      float crVal = fabs((offDiagSum / diag));
      float crThresh = crThreshold[0];
      if (crVal > crThresh) {
        cfMarking[row] = 1;
      } else {
        bool hasC = false;
        for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
          int k2 = (rs + j2);
          uint isStr = strongMask[k2];
          if (isStr == (uint)(1u)) {
            int nbr = ((int)(colIdx[k2]));
            int nbrM = cfMarking[nbr];
            if (nbrM == 1) {
              hasC = true;
            }
          }
        }
        if (hasC) {
          cfMarking[row] = -1;
        }
      }
    }
  }
}
