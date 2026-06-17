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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_compute_cluster_stability_f32(device const float* point_lambdas [[buffer(0)]], device const int* cluster_ids [[buffer(1)]], device const uint* n_edges [[buffer(2)]], device const uint* n_clusters [[buffer(3)]], device float* stabilities [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nEdges = n_edges[0];
  uint nClusters = n_clusters[0];
  bool inBounds = (gid < nClusters);
  if (inBounds) {
    float sumLambdas = as_type<float>(0x00000000u);
    int targetCluster = ((int)(gid));
    for (int e = 0; e < (int)(nEdges); ++e) {
      uint eU32 = ((uint)(e));
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
