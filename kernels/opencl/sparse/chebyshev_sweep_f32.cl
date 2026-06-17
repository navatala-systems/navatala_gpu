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

__kernel void navatala_sparse_chebyshev_sweep_f32(__global const float* r, __global const float* d, __global const float* coeffA, __global const float* coeffB, __global const uint* n, __global float* x, __global float* dNew) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if ((i < N)) {
    float cA = coeffA[0];
    float cB = coeffB[0];
    float ri = r[i];
    float di = d[i];
    float xi = x[i];
    float corr = ((cA * ri) + (cB * di));
    x[i] = (xi + corr);
    dNew[i] = corr;
  }
}
