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

kernel void navatala_sparse_spgemm_symbolic_row_nnz(device const uint* ArowPtr [[buffer(0)]], device const uint* AcolIdx [[buffer(1)]], device const uint* BrowPtr [[buffer(2)]], device const uint* BcolIdx [[buffer(3)]], device const uint* nRowsA [[buffer(4)]], device uint* CrowNnz [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRowsA[0]));
  if ((gid < N)) {
    int ars = ((int)(ArowPtr[gid]));
    int are = ((int)(ArowPtr[(gid + 1)]));
    int count = 0;
    for (int j = 0; j < (int)((are - ars)); ++j) {
      int ak = (ars + j);
      int bCol = ((int)(AcolIdx[ak]));
      int brs = ((int)(BrowPtr[bCol]));
      int bre = ((int)(BrowPtr[(bCol + 1)]));
      count = (count + (bre - brs));
    }
    CrowNnz[gid] = ((uint)(count));
  }
}
