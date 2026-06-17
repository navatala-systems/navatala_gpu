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

__kernel void navatala_dataframe_range_partition_bounds(__global const int* sortedData, __global const uint* dataCount, __global const uint* numPartitions, __global int* bounds) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = dataCount[(uint)(0u)];
  uint numParts = numPartitions[(uint)(0u)];
  uint numBounds = (numParts - (uint)(1u));
  bool inBounds = (gid < numBounds);
  if (inBounds) {
    uint boundIdx = (gid + (uint)(1u));
    uint sampleIdx = ((boundIdx * n) / numParts);
    uint maxIdx = (n - (uint)(1u));
    uint clampedIdx = (((sampleIdx < maxIdx)) ? (sampleIdx) : (maxIdx));
    int boundValue = sortedData[clampedIdx];
    bounds[gid] = boundValue;
  }
}
