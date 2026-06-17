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

__kernel void navatala_dataframe_murmur3_hash32_i64(__global const long* keys, __global const uint* seed, __global const uint* count, __global uint* hashes) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint s = seed[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    long key = keys[gid];
    uint kLow = ((uint)(key));
    uint kHigh = ((uint)((key >> 32)));
    uint c1 = (uint)(3432918353u);
    uint c2 = (uint)(461845907u);
    uint k1a = (kLow * c1);
    uint k1b = ((k1a << (uint)(15u)) | (k1a >> (uint)(17u)));
    uint k1c = (k1b * c2);
    uint h1a = (s ^ k1c);
    uint h1b = ((h1a << (uint)(13u)) | (h1a >> (uint)(19u)));
    uint h1c = ((h1b * (uint)(5u)) + (uint)(3864292196u));
    uint k2a = (kHigh * c1);
    uint k2b = ((k2a << (uint)(15u)) | (k2a >> (uint)(17u)));
    uint k2c = (k2b * c2);
    uint h2a = (h1c ^ k2c);
    uint h2b = ((h2a << (uint)(13u)) | (h2a >> (uint)(19u)));
    uint h2c = ((h2b * (uint)(5u)) + (uint)(3864292196u));
    uint h3 = (h2c ^ (uint)(8u));
    uint h4 = (h3 ^ (h3 >> (uint)(16u)));
    uint h5 = (h4 * (uint)(2246822507u));
    uint h6 = (h5 ^ (h5 >> (uint)(13u)));
    uint h7 = (h6 * (uint)(3266489909u));
    uint hash = (h7 ^ (h7 >> (uint)(16u)));
    hashes[gid] = hash;
  }
}
