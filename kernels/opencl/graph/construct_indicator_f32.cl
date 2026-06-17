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

__kernel void navatala_graph_construct_indicator_f32(__global const uint* clusters, __global const uint* clusterIndex, __global const uint* n, __global float* indicator) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint targetCluster = clusterIndex[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint nodeCluster = clusters[gid];
    bool isTarget = (nodeCluster == targetCluster);
    float indicatorVal = ((isTarget) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
    indicator[gid] = indicatorVal;
  }
}
