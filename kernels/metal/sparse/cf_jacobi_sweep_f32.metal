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

kernel void navatala_sparse_cf_jacobi_sweep_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* diag [[buffer(3)]], device const float* b [[buffer(4)]], device const float* x [[buffer(5)]], device const int* cfMarking [[buffer(6)]], device const float* omegaC [[buffer(7)]], device const float* omegaF [[buffer(8)]], device const uint* nRows [[buffer(9)]], device float* xNew [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float offDiag = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != row) {
        float a = values[k];
        float xj = x[col];
        offDiag = (offDiag + (a * xj));
      }
    }
    int cf = cfMarking[row];
    float om = (((cf == 1)) ? (omegaC[0]) : (omegaF[0]));
    float xr = x[row];
    float br = b[row];
    float dr = diag[row];
    xNew[row] = (((as_type<float>(0x3f800000u) - om) * xr) + (om * ((br - offDiag) / dr)));
  }
}
