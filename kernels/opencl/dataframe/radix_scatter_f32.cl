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

__kernel void navatala_dataframe_radix_scatter_f32(__global const float* _input, __global const uint* count, __global const uint* bitOffset, __global const uint* prefixSums, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint bits = bitOffset[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float val = _input[gid];
    float scaled = (val * as_float(0x4b800000u));
    int sortKeyRaw = ((int)(scaled));
    uint sortKey = ((uint)((sortKeyRaw + 2147483648)));
    uint digit = ((sortKey >> bits) & (uint)(255u));
    uint destBase = prefixSums[digit];
    _output[destBase] = val;
  }
}
