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

__kernel void navatala_dataframe_hash_partition_i32(__global const int* keys, __global const uint* numPartitions, __global const uint* count, __global uint* partitionIds) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint numParts = numPartitions[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    int key = keys[gid];
    uint k = ((uint)(key));
    uint h1 = (k ^ (k >> (uint)(16u)));
    uint h2 = (h1 * (uint)(2246822507u));
    uint h3 = (h2 ^ (h2 >> (uint)(13u)));
    uint h4 = (h3 * (uint)(3266489909u));
    uint hash = (h4 ^ (h4 >> (uint)(16u)));
    uint partId = (hash % numParts);
    partitionIds[gid] = partId;
  }
}
