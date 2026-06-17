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

__kernel void navatala_vector_search_merge_sorted_f32(__global const float* a, __global const float* b, __global const uint* a_indices, __global const uint* b_indices, __global const uint* n_a, __global const uint* n_b, __global float* merged, __global uint* merged_indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint na = n_a[0];
  uint nb = n_b[0];
  uint total = (na + nb);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint lo = (uint)(0u);
    uint hi = gid;
    uint hi_clamped = (((gid > na)) ? (na) : (gid));
    hi = hi_clamped;
    for (int __iter = 0; __iter < 32; ++__iter) {
      if (!((lo < hi))) break;
      uint lo_v = lo;
      uint hi_v = hi;
      uint mid = (lo_v + ((hi_v - lo_v) / (uint)(2u)));
      uint j = (gid - mid);
      float a_val = (((mid < na)) ? (a[mid]) : (as_float(0x7e967699u)));
      float b_prev = (((j > (uint)(0u))) ? (b[(j - (uint)(1u))]) : (as_float(0xfe967699u)));
      bool go_higher = (a_val < b_prev);
      if (go_higher) {
        lo = (mid + (uint)(1u));
      } else {
        hi = mid;
      }
    }
    uint i = lo;
    uint j = (gid - i);
    bool a_available = (i < na);
    bool b_available = (j < nb);
    float a_val = ((a_available) ? (a[i]) : (as_float(0x7e967699u)));
    float b_val = ((b_available) ? (b[j]) : (as_float(0x7e967699u)));
    bool pick_a = (a_val <= b_val);
    float out_val = ((pick_a) ? (a_val) : (b_val));
    uint out_idx = ((pick_a) ? (a_indices[i]) : (b_indices[j]));
    merged[gid] = out_val;
    merged_indices[gid] = out_idx;
  }
}
