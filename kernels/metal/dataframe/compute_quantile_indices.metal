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

kernel void navatala_dataframe_compute_quantile_indices(device const uint* quantile [[buffer(0)]], device const uint* count [[buffer(1)]], device uint* floorIdx [[buffer(2)]], device uint* ceilIdx [[buffer(3)]], device uint* fraction [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint q = quantile[0u];
  uint n = count[0u];
  uint nMinus1 = (n - 1u);
  uint posScaled = (q * nMinus1);
  uint floor = (posScaled / 1000u);
  uint ceilRaw = (floor + 1u);
  uint ceil = (((ceilRaw < n)) ? (ceilRaw) : (nMinus1));
  uint frac = (posScaled % 1000u);
  floorIdx[0u] = floor;
  ceilIdx[0u] = ceil;
  fraction[0u] = frac;
}
