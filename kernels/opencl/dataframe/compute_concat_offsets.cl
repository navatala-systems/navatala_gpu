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

__kernel void navatala_dataframe_compute_concat_offsets(__global const uint* lengths, __global const uint* numColumns, __global uint* offsets) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = numColumns[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    bool isFirst = (gid == (uint)(0u));
    uint len0 = (((n >= (uint)(1u))) ? (lengths[(uint)(0u)]) : ((uint)(0u)));
    uint len1 = (((n >= (uint)(2u))) ? (lengths[(uint)(1u)]) : ((uint)(0u)));
    uint len2 = (((n >= (uint)(3u))) ? (lengths[(uint)(2u)]) : ((uint)(0u)));
    uint len3 = (((n >= (uint)(4u))) ? (lengths[(uint)(3u)]) : ((uint)(0u)));
    uint prefix0 = (uint)(0u);
    uint prefix1 = len0;
    uint prefix2 = (prefix1 + len1);
    uint prefix3 = (prefix2 + len2);
    bool isIdx0 = (gid == (uint)(0u));
    bool isIdx1 = (gid == (uint)(1u));
    bool isIdx2 = (gid == (uint)(2u));
    bool isIdx3 = (gid == (uint)(3u));
    uint result = ((isIdx0) ? (prefix0) : (((isIdx1) ? (prefix1) : (((isIdx2) ? (prefix2) : (((isIdx3) ? (prefix3) : ((uint)(0u)))))))));
    offsets[gid] = result;
  }
}
