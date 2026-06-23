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

__kernel void navatala_sparse_jacobi_l1_sweep_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const float* b, __global const float* x, __global const float* omega, __global const uint* nRows, __global float* xNew) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float offDiag = as_float(0x00000000u);
    float l1Norm = as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float a = values[k];
      l1Norm = (l1Norm + fabs(a));
      if (col != row) {
        float xj = x[col];
        offDiag = (offDiag + (a * xj));
      }
    }
    float om = omega[0];
    float xr = x[row];
    float br = b[row];
    xNew[row] = (xr + (om * ((br - offDiag) / l1Norm)));
  }
}
