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

kernel void navatala_cfd_dic_build_reciprocal_d(device const float* diag [[buffer(0)]], device const float* upper [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* neighbour [[buffer(3)]], device const int* counts [[buffer(4)]], device float* rD [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  for (int c = 0; c < (int)(counts[0]); ++c) {
    rD[c] = diag[c];
  }
  for (int f = 0; f < (int)(counts[2]); ++f) {
    int o = owner[f];
    int n = neighbour[f];
    float u = upper[f];
    float rdo = rD[o];
    float rdn = rD[n];
    rD[n] = (rdn - ((u * u) / rdo));
  }
  for (int c2 = 0; c2 < (int)(counts[0]); ++c2) {
    float v = rD[c2];
    if ((v != as_type<float>(0x00000000u))) {
      rD[c2] = (as_type<float>(0x3f800000u) / v);
    } else {
      rD[c2] = as_type<float>(0x00000000u);
    }
  }
}
