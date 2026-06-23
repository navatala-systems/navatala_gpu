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

kernel void navatala_sparse_csr_to_ell_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* nRows [[buffer(3)]], device const uint* maxNnzPerRow [[buffer(4)]], device uint* ellColIdx [[buffer(5)]], device float* ellValues [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    int nnz = (re - rs);
    int maxNnz = ((int)(maxNnzPerRow[0]));
    for (int j = 0; j < (int)(maxNnz); ++j) {
      if (j < nnz) {
        int k = (rs + j);
        ellColIdx[((gid * maxNnz) + j)] = colIdx[k];
        ellValues[((gid * maxNnz) + j)] = values[k];
      } else {
        ellColIdx[((gid * maxNnz) + j)] = 4294967295u;
        ellValues[((gid * maxNnz) + j)] = as_type<float>(0x00000000u);
      }
    }
  }
}
