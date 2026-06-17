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

__kernel void navatala_sparse_polynomial_sweep_f32(__global const float* Ax, __global const float* x, __global const float* b, __global const float* coeffs, __global const uint* degree, __global const uint* n, __global float* xNew) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if ((i < N)) {
    float ri = (b[i] - Ax[i]);
    float xi = x[i];
    int deg = ((int)(degree[0]));
    float poly = as_float(0x00000000u);
    float power = as_float(0x3f800000u);
    for (int k = 0; k < (int)(deg); ++k) {
      float ck = coeffs[k];
      poly = (poly + (ck * power));
      power = (power * ri);
    }
    xNew[i] = (xi + poly);
  }
}
