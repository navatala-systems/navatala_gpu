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
extern "C" __global__ void navatala_sparse_dense_l_u_solve_f64(const double* LU, const unsigned int* pivots, const double* b, const unsigned int* n, double* x) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  for (int iCpy = 0; iCpy < (int)(N); ++iCpy) {
    x[iCpy] = b[iCpy];
  }
  for (int k = 0; k < (int)(N); ++k) {
    int pk = ((int)(pivots[k]));
    if (pk != k) {
      double tmp = x[k];
      x[k] = x[pk];
      x[pk] = tmp;
    }
  }
  for (int i2 = 0; i2 < (int)(N); ++i2) {
    double sum = x[i2];
    for (int j = 0; j < (int)(i2); ++j) {
      double lij = LU[((i2 * N) + j)];
      sum = (sum - (lij * x[j]));
    }
    x[i2] = sum;
  }
  for (int ii = 0; ii < (int)(N); ++ii) {
    int i3 = ((N - 1) - ii);
    double sum2 = x[i3];
    int start = (i3 + 1);
    for (int j2 = 0; j2 < (int)((N - start)); ++j2) {
      int c = (start + j2);
      double uij = LU[((i3 * N) + c)];
      sum2 = (sum2 - (uij * x[c]));
    }
    double diag = LU[((i3 * N) + i3)];
    x[i3] = (sum2 / diag);
  }
}
