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

__kernel void navatala_sparse_lobpcg_rayleigh_ritz_f32(__global const float* X, __global const float* AX, __global const uint* n, __global const uint* k, __global float* XtAX, __global float* XtX) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  int kVal = ((int)(k[0]));
  if (gid < (kVal * kVal)) {
    int row = (gid / kVal);
    int col = (gid % kVal);
    float dotAX = as_float(0x00000000u);
    float dotXX = as_float(0x00000000u);
    for (int i = 0; i < (int)(N); ++i) {
      float xi = X[((col * N) + i)];
      float axi = AX[((row * N) + i)];
      dotAX = (dotAX + (xi * axi));
      float xj = X[((row * N) + i)];
      dotXX = (dotXX + (xi * xj));
    }
    XtAX[gid] = dotAX;
    XtX[gid] = dotXX;
  }
}
