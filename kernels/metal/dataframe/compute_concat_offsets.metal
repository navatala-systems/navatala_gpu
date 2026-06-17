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

kernel void navatala_dataframe_compute_concat_offsets(device const uint* lengths [[buffer(0)]], device const uint* numColumns [[buffer(1)]], device uint* offsets [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = numColumns[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    bool isFirst = (gid == 0u);
    uint len0 = (((n >= 1u)) ? (lengths[0u]) : (0u));
    uint len1 = (((n >= 2u)) ? (lengths[1u]) : (0u));
    uint len2 = (((n >= 3u)) ? (lengths[2u]) : (0u));
    uint len3 = (((n >= 4u)) ? (lengths[3u]) : (0u));
    uint prefix0 = 0u;
    uint prefix1 = len0;
    uint prefix2 = (prefix1 + len1);
    uint prefix3 = (prefix2 + len2);
    bool isIdx0 = (gid == 0u);
    bool isIdx1 = (gid == 1u);
    bool isIdx2 = (gid == 2u);
    bool isIdx3 = (gid == 3u);
    uint result = ((isIdx0) ? (prefix0) : (((isIdx1) ? (prefix1) : (((isIdx2) ? (prefix2) : (((isIdx3) ? (prefix3) : (0u))))))));
    offsets[gid] = result;
  }
}
