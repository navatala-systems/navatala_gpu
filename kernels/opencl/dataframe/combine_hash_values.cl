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

__kernel void navatala_dataframe_combine_hash_values(__global const uint* hashes1, __global const uint* hashes2, __global const uint* count, __global uint* combined) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint h1 = hashes1[gid];
    uint h2 = hashes2[gid];
    uint golden = (uint)(2654435769u);
    uint h2Left = (h2 << (uint)(5u));
    uint h2Right = (h2 >> (uint)(27u));
    uint h2Rotated = (h2Left | h2Right);
    uint h2WithGolden = (h2Rotated + golden);
    uint result = (h1 ^ h2WithGolden);
    combined[gid] = result;
  }
}
