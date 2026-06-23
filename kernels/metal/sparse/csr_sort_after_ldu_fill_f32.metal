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

kernel void navatala_sparse_csr_sort_after_ldu_fill_f32(device const uint* rowPtr [[buffer(0)]], device const uint* nRows [[buffer(1)]], device uint* colInd [[buffer(2)]], device float* values [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int start = ((int)(rowPtr[row]));
    int end = ((int)(rowPtr[(row + 1)]));
    int count = (end - start);
    for (int i = 0; i < (int)(count); ++i) {
      int pos = (start + i);
      uint keyCol = colInd[pos];
      float keyVal = values[pos];
      int j = i;
      for (int __iter = 0; __iter < 64; ++__iter) {
        if (!((j > 0) && (colInd[((start + j) - 1)] > keyCol))) break;
        colInd[(start + j)] = colInd[((start + j) - 1)];
        values[(start + j)] = values[((start + j) - 1)];
        j = (j - 1);
      }
      colInd[(start + j)] = keyCol;
      values[(start + j)] = keyVal;
    }
  }
}
