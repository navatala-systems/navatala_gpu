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
extern "C" __global__ void navatala_sparse_kpz_polynomial_sweep_f64(const double* Ax, const double* x, const double* b, const double* kpzCoeffs, const unsigned int* degree, const unsigned int* n, double* xNew) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(n[0]));
  if ((i < N)) {
    double ri = (b[i] - Ax[i]);
    double xi = x[i];
    int deg = ((int)(degree[0]));
    double poly = __longlong_as_double(0x0000000000000000ull);
    double power = __longlong_as_double(0x3ff0000000000000ull);
    for (int k = 0; k < (int)(deg); ++k) {
      double ck = kpzCoeffs[k];
      poly = (poly + (ck * power));
      power = (power * ri);
    }
    xNew[i] = (xi + poly);
  }
}
