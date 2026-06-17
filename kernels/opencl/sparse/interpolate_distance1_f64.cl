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
__kernel void navatala_sparse_interpolate_distance1_f64(__global const uint* rowPtr, __global const uint* colIdx, __global const double* values, __global const uint* strongMask, __global const int* cfMarking, __global const uint* ProwPtr, __global const uint* nRows, __global uint* PcolIdx, __global double* Pvalues) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    int mark = cfMarking[row];
    int pStart = ((int)(ProwPtr[row]));
    if ((mark == 1)) {
      PcolIdx[pStart] = ((uint)(row));
      Pvalues[pStart] = as_double(0x3ff0000000000000ul);
    } else {
      if ((mark == -1)) {
        double diag = as_double(0x0000000000000000ul);
        for (int jd = 0; jd < (int)((re - rs)); ++jd) {
          int kd = (rs + jd);
          int colD = ((int)(colIdx[kd]));
          if ((colD == row)) {
            diag = values[kd];
          }
        }
        int pos = 0;
        for (int jw = 0; jw < (int)((re - rs)); ++jw) {
          int kw = (rs + jw);
          uint isStrW = strongMask[kw];
          if ((isStrW == (uint)(1u))) {
            int colW = ((int)(colIdx[kw]));
            int markW = cfMarking[colW];
            if ((markW == 1)) {
              double aij = values[kw];
              double w = (-(aij / diag));
              PcolIdx[(pStart + pos)] = ((uint)(colW));
              Pvalues[(pStart + pos)] = w;
              pos = (pos + 1);
            }
          }
        }
      }
    }
  }
}
