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

kernel void navatala_sparse_spgemm_numeric_f32(device const uint* ArowPtr [[buffer(0)]], device const uint* AcolIdx [[buffer(1)]], device const float* Avalues [[buffer(2)]], device const uint* BrowPtr [[buffer(3)]], device const uint* BcolIdx [[buffer(4)]], device const float* Bvalues [[buffer(5)]], device const uint* CrowPtr [[buffer(6)]], device const uint* nRowsA [[buffer(7)]], device uint* CcolIdx [[buffer(8)]], device float* Cvalues [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRowsA[0]));
  if (gid < N) {
    int ars = ((int)(ArowPtr[gid]));
    int are = ((int)(ArowPtr[(gid + 1)]));
    int crs = ((int)(CrowPtr[gid]));
    int ci = 0;
    for (int j = 0; j < (int)((are - ars)); ++j) {
      int ak = (ars + j);
      float aVal = Avalues[ak];
      int bCol = ((int)(AcolIdx[ak]));
      int brs = ((int)(BrowPtr[bCol]));
      int bre = ((int)(BrowPtr[(bCol + 1)]));
      for (int jj = 0; jj < (int)((bre - brs)); ++jj) {
        int bk = (brs + jj);
        int tgt = (crs + ci);
        uint bColIdx = BcolIdx[bk];
        float bVal = Bvalues[bk];
        CcolIdx[tgt] = bColIdx;
        Cvalues[tgt] = (aVal * bVal);
        ci = (ci + 1);
      }
    }
  }
}
