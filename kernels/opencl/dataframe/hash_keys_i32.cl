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

__kernel void navatala_dataframe_hash_keys_i32(__global const int* keys, __global const uint* validity, __global const uint* count, __global const uint* tableSize, __global uint* hashes) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint tSize = tableSize[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & (uint)(1u));
    bool isValid = (validBit == (uint)(1u));
    int key = keys[gid];
    uint k1 = ((uint)(key));
    uint k2 = (k1 * (uint)(3432918353u));
    uint k3 = ((k2 << (uint)(15u)) | (k2 >> (uint)(17u)));
    uint k4 = (k3 * (uint)(461845907u));
    uint h1 = (k4 ^ (uint)(2246822507u));
    uint h2 = (h1 ^ (h1 >> (uint)(16u)));
    uint h3 = (h2 * (uint)(3266489909u));
    uint h4 = (h3 ^ (h3 >> (uint)(13u)));
    uint hashIdx = (h4 % tSize);
    uint outHash = ((isValid) ? (hashIdx) : ((uint)(2147483647u)));
    hashes[gid] = outHash;
  }
}
