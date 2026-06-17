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
__kernel void navatala_ml_compute_cluster_stability_f64(__global const double* point_lambdas, __global const int* cluster_ids, __global const uint* n_edges, __global const uint* n_clusters, __global double* stabilities) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nEdges = n_edges[0];
  uint nClusters = n_clusters[0];
  bool inBounds = (gid < nClusters);
  if (inBounds) {
    double sumLambdas = as_double(0x0000000000000000ul);
    int targetCluster = ((int)(gid));
    for (int e = 0; e < (int)(nEdges); ++e) {
      uint eU32 = ((uint)(e));
      int edgeCluster = cluster_ids[eU32];
      bool isTargetCluster = (edgeCluster == targetCluster);
      if (isTargetCluster) {
        double lambda = point_lambdas[eU32];
        double currSum = sumLambdas;
        double newSum = (currSum + lambda);
        sumLambdas = newSum;
      }
    }
    double finalStability = sumLambdas;
    stabilities[gid] = finalStability;
  }
}
