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

__kernel void navatala_dataframe_range_partition_assign(__global const int* keys, __global const int* bounds, __global const uint* numPartitions, __global const uint* count, __global uint* partitionIds) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint numParts = numPartitions[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    int key = keys[gid];
    uint numBounds = (numParts - (uint)(1u));
    int b0 = (((numBounds >= (uint)(1u))) ? (bounds[(uint)(0u)]) : (2147483647));
    int b1 = (((numBounds >= (uint)(2u))) ? (bounds[(uint)(1u)]) : (2147483647));
    int b2 = (((numBounds >= (uint)(3u))) ? (bounds[(uint)(2u)]) : (2147483647));
    int b3 = (((numBounds >= (uint)(4u))) ? (bounds[(uint)(3u)]) : (2147483647));
    uint ge0 = (((key >= b0)) ? ((uint)(1u)) : ((uint)(0u)));
    uint ge1 = (((key >= b1)) ? ((uint)(1u)) : ((uint)(0u)));
    uint ge2 = (((key >= b2)) ? ((uint)(1u)) : ((uint)(0u)));
    uint ge3 = (((key >= b3)) ? ((uint)(1u)) : ((uint)(0u)));
    uint partId = (ge0 + (ge1 + (ge2 + ge3)));
    uint maxPartId = (numParts - (uint)(1u));
    uint clampedPartId = (((partId < maxPartId)) ? (partId) : (maxPartId));
    partitionIds[gid] = clampedPartId;
  }
}
