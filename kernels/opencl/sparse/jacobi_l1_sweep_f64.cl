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
__kernel void navatala_sparse_jacobi_l1_sweep_f64(__global const uint* rowPtr, __global const uint* colIdx, __global const double* values, __global const double* b, __global const double* x, __global const double* omega, __global const uint* nRows, __global double* xNew) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    double offDiag = as_double(0x0000000000000000ul);
    double l1Norm = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      double a = values[k];
      l1Norm = (l1Norm + fabs(a));
      if ((col != row)) {
        double xj = x[col];
        offDiag = (offDiag + (a * xj));
      }
    }
    double om = omega[0];
    double xr = x[row];
    double br = b[row];
    xNew[row] = (xr + (om * ((br - offDiag) / l1Norm)));
  }
}
