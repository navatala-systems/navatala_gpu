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

kernel void navatala_vector_search_merge_sorted_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* a_indices [[buffer(2)]], device const uint* b_indices [[buffer(3)]], device const uint* n_a [[buffer(4)]], device const uint* n_b [[buffer(5)]], device float* merged [[buffer(6)]], device uint* merged_indices [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint na = n_a[0];
  uint nb = n_b[0];
  uint total = (na + nb);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint lo = 0u;
    uint hi = gid;
    uint hi_clamped = (((gid > na)) ? (na) : (gid));
    hi = hi_clamped;
    for (int __iter = 0; __iter < 32; ++__iter) {
      if (!((lo < hi))) break;
      uint lo_v = lo;
      uint hi_v = hi;
      uint mid = (lo_v + ((hi_v - lo_v) / 2u));
      uint j = (gid - mid);
      float a_val = (((mid < na)) ? (a[mid]) : (as_type<float>(0x7e967699u)));
      float b_prev = (((j > 0u)) ? (b[(j - 1u)]) : (as_type<float>(0xfe967699u)));
      bool go_higher = (a_val < b_prev);
      if (go_higher) {
        lo = (mid + 1u);
      } else {
        hi = mid;
      }
    }
    uint i = lo;
    uint j = (gid - i);
    bool a_available = (i < na);
    bool b_available = (j < nb);
    float a_val = ((a_available) ? (a[i]) : (as_type<float>(0x7e967699u)));
    float b_val = ((b_available) ? (b[j]) : (as_type<float>(0x7e967699u)));
    bool pick_a = (a_val <= b_val);
    float out_val = ((pick_a) ? (a_val) : (b_val));
    uint out_idx = ((pick_a) ? (a_indices[i]) : (b_indices[j]));
    merged[gid] = out_val;
    merged_indices[gid] = out_idx;
  }
}
