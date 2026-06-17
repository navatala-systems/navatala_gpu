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

__kernel void navatala_dataframe_radix_scatter_i64(__global const long* _input, __global const uint* count, __global const uint* bitOffset, __global const uint* prefixSums, __global long* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint bits = bitOffset[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    long val = _input[gid];
    long signFlipped = (val ^ 9223372036854775808);
    uint bitsU32 = bits;
    long shifted = (signFlipped >> ((long)(bitsU32)));
    uint digit = ((uint)((shifted & 255)));
    uint destBase = prefixSums[digit];
    _output[destBase] = val;
  }
}
