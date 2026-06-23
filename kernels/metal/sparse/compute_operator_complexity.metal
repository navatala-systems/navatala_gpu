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

kernel void navatala_sparse_compute_operator_complexity(device const uint* levelNnz [[buffer(0)]], device const uint* levelRows [[buffer(1)]], device const uint* nLevels [[buffer(2)]], device float* operatorComplexity [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    int nLev = ((int)(nLevels[0]));
    uint finestNnz = levelNnz[0];
    float total = as_type<float>(0x00000000u);
    for (int lev = 0; lev < (int)(nLev); ++lev) {
      uint nnzL = levelNnz[lev];
      total = (total + ((float)(nnzL)));
    }
    float oc = (total / ((float)(finestNnz)));
    operatorComplexity[0] = oc;
  }
}
