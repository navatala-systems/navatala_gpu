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

kernel void navatala_cfd_restrict_by_agglomeration_gather_f32(device const float* fineField [[buffer(0)]], device const uint* coarseToFineOffsets [[buffer(1)]], device const uint* coarseToFineList [[buffer(2)]], device const uint* nCoarseCells [[buffer(3)]], device float* coarseField [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int c = int(__gid.x);
  int N = ((int)(nCoarseCells[0]));
  if ((c < N)) {
    int begin = ((int)(coarseToFineOffsets[c]));
    int end = ((int)(coarseToFineOffsets[(c + 1)]));
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((end - begin)); ++j) {
      int pos = (begin + j);
      int fineIdx = ((int)(coarseToFineList[pos]));
      float v = fineField[fineIdx];
      sum = (sum + v);
    }
    coarseField[c] = sum;
  }
}
