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

__kernel void navatala_dataframe_murmur3_hash32_i32(__global const int* keys, __global const uint* seed, __global const uint* count, __global uint* hashes) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint s = seed[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    int key = keys[gid];
    uint k = ((uint)(key));
    uint c1 = (uint)(3432918353u);
    uint c2 = (uint)(461845907u);
    uint k1 = (k * c1);
    uint k2 = ((k1 << (uint)(15u)) | (k1 >> (uint)(17u)));
    uint k3 = (k2 * c2);
    uint h1 = (s ^ k3);
    uint h2 = ((h1 << (uint)(13u)) | (h1 >> (uint)(19u)));
    uint h3 = ((h2 * (uint)(5u)) + (uint)(3864292196u));
    uint h4 = (h3 ^ (uint)(4u));
    uint h5 = (h4 ^ (h4 >> (uint)(16u)));
    uint h6 = (h5 * (uint)(2246822507u));
    uint h7 = (h6 ^ (h6 >> (uint)(13u)));
    uint h8 = (h7 * (uint)(3266489909u));
    uint hash = (h8 ^ (h8 >> (uint)(16u)));
    hashes[gid] = hash;
  }
}
