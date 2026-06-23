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

kernel void navatala_sparse_reorder_c_s_r_by_color(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* reorderPerm [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* newRowPtr [[buffer(5)]], device uint* newColIdx [[buffer(6)]], device float* newValues [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int origRow = ((int)(reorderPerm[gid]));
    int rs = ((int)(rowPtr[origRow]));
    int re = ((int)(rowPtr[(origRow + 1)]));
    int nnz = (re - rs);
    newRowPtr[gid] = ((uint)(nnz));
    for (int j = 0; j < (int)(nnz); ++j) {
      int k = (rs + j);
      newColIdx[(gid + j)] = colIdx[k];
      newValues[(gid + j)] = values[k];
    }
  }
}
