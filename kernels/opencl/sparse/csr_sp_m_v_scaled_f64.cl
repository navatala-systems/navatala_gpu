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
__kernel void navatala_sparse_csr_sp_m_v_scaled_f64(__global const uint* rowPtr, __global const uint* colIdx, __global const double* values, __global const double* x, __global const double* alpha, __global const double* beta, __global const uint* nRows, __global double* y) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rowStart = ((int)(rowPtr[row]));
    int rowEnd = ((int)(rowPtr[(row + 1)]));
    double acc = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      int k = (rowStart + j);
      int col = ((int)(colIdx[k]));
      double aij = values[k];
      double xj = x[col];
      acc = (acc + (aij * xj));
    }
    double alphaVal = alpha[0];
    double betaVal = beta[0];
    double yi = y[row];
    y[row] = ((alphaVal * acc) + (betaVal * yi));
  }
}
