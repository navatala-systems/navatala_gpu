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

__kernel void navatala_sparse_dense_l_u_f32(__global const uint* n, __global float* A, __global uint* pivots) {
  int gid0 = (int)get_global_id(0);
  int N = ((int)(n[0]));
  for (int k = 0; k < (int)(N); ++k) {
    float maxVal = A[((k * N) + k)];
    int maxRow = k;
    for (int i = 0; i < (int)((N - k)); ++i) {
      int r = (k + i);
      float v = A[((r * N) + k)];
      if ((fabs(v) > fabs(maxVal))) {
        maxVal = v;
        maxRow = r;
      }
    }
    pivots[k] = ((uint)(maxRow));
    if ((maxRow != k)) {
      for (int j = 0; j < (int)(N); ++j) {
        float a = A[((k * N) + j)];
        float b = A[((maxRow * N) + j)];
        A[((k * N) + j)] = b;
        A[((maxRow * N) + j)] = a;
      }
    }
    float pivot = A[((k * N) + k)];
    for (int i2 = 0; i2 < (int)((N - (k + 1))); ++i2) {
      int r2 = ((k + 1) + i2);
      float factor = (A[((r2 * N) + k)] / pivot);
      A[((r2 * N) + k)] = factor;
      for (int j2 = 0; j2 < (int)((N - (k + 1))); ++j2) {
        int c = ((k + 1) + j2);
        float old = A[((r2 * N) + c)];
        A[((r2 * N) + c)] = (old - (factor * A[((k * N) + c)]));
      }
    }
  }
}
