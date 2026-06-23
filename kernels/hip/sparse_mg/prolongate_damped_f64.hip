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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_sparse_mg_prolongate_damped_f64(const unsigned int* ProwPtr, const unsigned int* PcolIdx, const double* Pvalues, const double* eCoarse, const double* omega, const unsigned int* nFine, double* xFine) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nFine[0]));
  if (row < N) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    double pe = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(PcolIdx[k]));
      double pv = Pvalues[k];
      double ec = eCoarse[col];
      pe = (pe + (pv * ec));
    }
    double om = omega[0];
    double xold = xFine[row];
    xFine[row] = (xold + (om * pe));
  }
}
