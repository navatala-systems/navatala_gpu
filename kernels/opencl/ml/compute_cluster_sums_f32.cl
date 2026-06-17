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

__kernel void navatala_ml_compute_cluster_sums_f32(__global const float* data, __global const uint* assignments, __global const uint* n, __global const uint* k, __global const uint* d, __global float* clusterSums, __global uint* clusterCounts) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint dVal = d[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint cluster = assignments[gid];
    atomic_add(&clusterCounts[cluster], (uint)(1u));
    uint dataBase = (gid * dVal);
    uint sumBase = (cluster * dVal);
    for (int j = 0; j < (int)(dVal); ++j) {
      uint jU32 = ((uint)(j));
      uint dataIdx = (dataBase + jU32);
      uint sumIdx = (sumBase + jU32);
      float dataVal = data[dataIdx];
      atomic_add(&clusterSums[sumIdx], dataVal);
    }
  }
}
