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
extern "C" __global__ void navatala_sparse_lobpcg_rayleigh_ritz_f64(const double* X, const double* AX, const unsigned int* n, const unsigned int* k, double* XtAX, double* XtX) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  int kVal = ((int)(k[0]));
  if (gid < (kVal * kVal)) {
    int row = (gid / kVal);
    int col = (gid % kVal);
    double dotAX = __longlong_as_double(0x0000000000000000ull);
    double dotXX = __longlong_as_double(0x0000000000000000ull);
    for (int i = 0; i < (int)(N); ++i) {
      double xi = X[((col * N) + i)];
      double axi = AX[((row * N) + i)];
      dotAX = (dotAX + (xi * axi));
      double xj = X[((row * N) + i)];
      dotXX = (dotXX + (xi * xj));
    }
    XtAX[gid] = dotAX;
    XtX[gid] = dotXX;
  }
}
