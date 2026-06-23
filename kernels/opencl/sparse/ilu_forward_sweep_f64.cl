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
__kernel void navatala_sparse_ilu_forward_sweep_f64(__global const uint* rowPtr, __global const uint* colIdx, __global const double* luValues, __global const double* b, __global const uint* colorOffsets, __global const uint* nRowsThisColor, __global double* y) {
  int gid0 = (int)get_global_id(0);
  int localIdx = (int)(get_global_id(0));
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int offset = ((int)(colorOffsets[0]));
    int row = (offset + localIdx);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    double bi = b[row];
    double sum = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col < row) {
        double a = luValues[k];
        double yj = y[col];
        sum = (sum + (a * yj));
      }
    }
    y[row] = (bi - sum);
  }
}
