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
extern "C" __global__ void navatala_ml_compute_cluster_stability_f32(const float* point_lambdas, const int* cluster_ids, const unsigned int* n_edges, const unsigned int* n_clusters, float* stabilities) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nEdges = n_edges[0];
  unsigned int nClusters = n_clusters[0];
  bool inBounds = (gid < nClusters);
  if (inBounds) {
    float sumLambdas = __uint_as_float(0x00000000u);
    int targetCluster = ((int)(gid));
    for (int e = 0; e < (int)(nEdges); ++e) {
      unsigned int eU32 = ((unsigned int)(e));
      int edgeCluster = cluster_ids[eU32];
      bool isTargetCluster = (edgeCluster == targetCluster);
      if (isTargetCluster) {
        float lambda = point_lambdas[eU32];
        float currSum = sumLambdas;
        float newSum = (currSum + lambda);
        sumLambdas = newSum;
      }
    }
    float finalStability = sumLambdas;
    stabilities[gid] = finalStability;
  }
}
