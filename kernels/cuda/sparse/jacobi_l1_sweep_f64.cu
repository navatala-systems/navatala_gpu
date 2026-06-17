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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_sparse_jacobi_l1_sweep_f64(const unsigned int* rowPtr, const unsigned int* colIdx, const double* values, const double* b, const double* x, const double* omega, const unsigned int* nRows, double* xNew) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    double offDiag = __longlong_as_double(0x0000000000000000ull);
    double l1Norm = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      double a = values[k];
      l1Norm = (l1Norm + abs(a));
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
