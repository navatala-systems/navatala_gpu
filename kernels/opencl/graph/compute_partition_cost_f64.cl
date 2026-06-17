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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_compute_partition_cost_f64(__global const double* edgeCutPerCluster, __global const uint* clusterSizes, __global const uint* numClusters, __global double* cost) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numClustersVal = numClusters[0];
  bool inBounds = (gid < numClustersVal);
  if (inBounds) {
    double clusterEdgeCut = edgeCutPerCluster[gid];
    uint clusterSize = clusterSizes[gid];
    bool sizeIsZero = (clusterSize == (uint)(0u));
    double clusterSizeF = ((double)(clusterSize));
    double normalizedCost = ((sizeIsZero) ? (as_double(0x0000000000000000ul)) : ((clusterEdgeCut / clusterSizeF)));
    bool hasContribution = (normalizedCost > as_double(0x0000000000000000ul));
    if (hasContribution) {
      atomic_add(&cost[0], normalizedCost);
    }
  }
}
