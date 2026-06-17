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

__kernel void navatala_dataframe_group_by_nunique_helper_i64(__global const long* data, __global const uint* validity, __global const uint* groupIds, __global const uint* count, __global uint* hashOutput, __global uint* validOutput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & (uint)(1u));
    bool isValid = (validBit == (uint)(1u));
    long val = data[gid];
    uint groupId = groupIds[gid];
    uint kLow = ((uint)(val));
    uint kHigh = ((uint)((val >> 32)));
    uint c1 = (uint)(3432918353u);
    uint c2 = (uint)(461845907u);
    uint k1a = (kLow * c1);
    uint k1b = ((k1a << (uint)(15u)) | (k1a >> (uint)(17u)));
    uint k1c = (k1b * c2);
    uint k2a = (kHigh * c1);
    uint k2b = ((k2a << (uint)(15u)) | (k2a >> (uint)(17u)));
    uint k2c = (k2b * c2);
    uint h1 = (groupId ^ k1c);
    uint h2 = (h1 ^ k2c);
    uint h3 = ((h2 << (uint)(13u)) | (h2 >> (uint)(19u)));
    uint h4 = ((h3 * (uint)(5u)) + (uint)(3864292196u));
    uint h5 = (h4 ^ (uint)(8u));
    uint h6 = (h5 ^ (h5 >> (uint)(16u)));
    uint h7 = (h6 * (uint)(2246822507u));
    uint h8 = (h7 ^ (h7 >> (uint)(13u)));
    uint h9 = (h8 * (uint)(3266489909u));
    uint hash = (h9 ^ (h9 >> (uint)(16u)));
    uint outHash = ((isValid) ? (hash) : ((uint)(0u)));
    hashOutput[gid] = outHash;
    uint outBit = (validBit << bitIdx);
    atomic_or(&validOutput[wordIdx], outBit);
  }
}
