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
extern "C" __global__ void navatala_sparse_cf_jacobi_sweep_f64(const unsigned int* rowPtr, const unsigned int* colIdx, const double* values, const double* diag, const double* b, const double* x, const int* cfMarking, const double* omegaC, const double* omegaF, const unsigned int* nRows, double* xNew) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    double offDiag = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != row) {
        double a = values[k];
        double xj = x[col];
        offDiag = (offDiag + (a * xj));
      }
    }
    int cf = cfMarking[row];
    double om = (((cf == 1)) ? (omegaC[0]) : (omegaF[0]));
    double xr = x[row];
    double br = b[row];
    double dr = diag[row];
    xNew[row] = (((__longlong_as_double(0x3ff0000000000000ull) - om) * xr) + (om * ((br - offDiag) / dr)));
  }
}
