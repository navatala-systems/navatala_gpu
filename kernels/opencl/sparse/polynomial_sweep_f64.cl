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
__kernel void navatala_sparse_polynomial_sweep_f64(__global const double* Ax, __global const double* x, __global const double* b, __global const double* coeffs, __global const uint* degree, __global const uint* n, __global double* xNew) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if (i < N) {
    double ri = (b[i] - Ax[i]);
    double xi = x[i];
    int deg = ((int)(degree[0]));
    double poly = as_double(0x0000000000000000ul);
    double power = as_double(0x3ff0000000000000ul);
    for (int k = 0; k < (int)(deg); ++k) {
      double ck = coeffs[k];
      poly = (poly + (ck * power));
      power = (power * ri);
    }
    xNew[i] = (xi + poly);
  }
}
