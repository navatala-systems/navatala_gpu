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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_compute_partition_cost_f64(const double* edgeCutPerCluster, const unsigned int* clusterSizes, const unsigned int* numClusters, double* cost) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numClustersVal = numClusters[0];
  bool inBounds = (gid < numClustersVal);
  if (inBounds) {
    double clusterEdgeCut = edgeCutPerCluster[gid];
    unsigned int clusterSize = clusterSizes[gid];
    bool sizeIsZero = (clusterSize == 0u);
    double clusterSizeF = ((double)(clusterSize));
    double normalizedCost = ((sizeIsZero) ? (__longlong_as_double(0x0000000000000000ull)) : ((clusterEdgeCut / clusterSizeF)));
    bool hasContribution = (normalizedCost > __longlong_as_double(0x0000000000000000ull));
    if (hasContribution) {
      atomicAdd(&cost[0], normalizedCost);
    }
  }
}
