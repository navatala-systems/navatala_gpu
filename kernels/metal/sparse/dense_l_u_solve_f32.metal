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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_dense_l_u_solve_f32(device const float* LU [[buffer(0)]], device const uint* pivots [[buffer(1)]], device const float* b [[buffer(2)]], device const uint* n [[buffer(3)]], device float* x [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int N = ((int)(n[0]));
  for (int iCpy = 0; iCpy < (int)(N); ++iCpy) {
    x[iCpy] = b[iCpy];
  }
  for (int k = 0; k < (int)(N); ++k) {
    int pk = ((int)(pivots[k]));
    if ((pk != k)) {
      float tmp = x[k];
      x[k] = x[pk];
      x[pk] = tmp;
    }
  }
  for (int i2 = 0; i2 < (int)(N); ++i2) {
    float sum = x[i2];
    for (int j = 0; j < (int)(i2); ++j) {
      float lij = LU[((i2 * N) + j)];
      sum = (sum - (lij * x[j]));
    }
    x[i2] = sum;
  }
  for (int ii = 0; ii < (int)(N); ++ii) {
    int i3 = ((N - 1) - ii);
    float sum2 = x[i3];
    int start = (i3 + 1);
    for (int j2 = 0; j2 < (int)((N - start)); ++j2) {
      int c = (start + j2);
      float uij = LU[((i3 * N) + c)];
      sum2 = (sum2 - (uij * x[c]));
    }
    float diag = LU[((i3 * N) + i3)];
    x[i3] = (sum2 / diag);
  }
}
