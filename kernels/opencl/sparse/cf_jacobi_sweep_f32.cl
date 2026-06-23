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

__kernel void navatala_sparse_cf_jacobi_sweep_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const float* diag, __global const float* b, __global const float* x, __global const int* cfMarking, __global const float* omegaC, __global const float* omegaF, __global const uint* nRows, __global float* xNew) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float offDiag = as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != row) {
        float a = values[k];
        float xj = x[col];
        offDiag = (offDiag + (a * xj));
      }
    }
    int cf = cfMarking[row];
    float om = (((cf == 1)) ? (omegaC[0]) : (omegaF[0]));
    float xr = x[row];
    float br = b[row];
    float dr = diag[row];
    xNew[row] = (((as_float(0x3f800000u) - om) * xr) + (om * ((br - offDiag) / dr)));
  }
}
