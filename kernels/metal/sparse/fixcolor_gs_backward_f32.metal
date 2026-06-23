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

kernel void navatala_sparse_fixcolor_gs_backward_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* diag [[buffer(3)]], device const float* b [[buffer(4)]], device const uint* reorderPerm [[buffer(5)]], device const uint* colorOffsets [[buffer(6)]], device const uint* nRowsThisColor [[buffer(7)]], device float* x [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int localIdx = int(__gid.x);
  int N = ((int)(nRowsThisColor[0]));
  if (localIdx < N) {
    int start = ((int)(colorOffsets[0]));
    int permIdx = (start + localIdx);
    int row = ((int)(reorderPerm[permIdx]));
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if (col != row) {
        float a = values[k];
        float xj = x[col];
        sum = (sum + (a * xj));
      }
    }
    float bi = b[row];
    float di = diag[row];
    x[row] = ((bi - sum) / di);
  }
}
