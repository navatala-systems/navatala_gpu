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

kernel void navatala_ml_extract_cluster_labels_f32(device const uint* selected_clusters [[buffer(0)]], device const int* point_cluster_map [[buffer(1)]], device const int* cluster_parent_map [[buffer(2)]], device const uint* n_samples [[buffer(3)]], device const uint* n_clusters [[buffer(4)]], device int* labels [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nClusters = n_clusters[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    int initialCluster = point_cluster_map[gid];
    int currentCluster = initialCluster;
    int foundLabel = -1;
    for (int level = 0; level < (int)(64u); ++level) {
      int currClust = currentCluster;
      bool isValidCluster = (currClust >= 0);
      if (isValidCluster) {
        uint currClustU32 = ((uint)(currClust));
        bool inClusterBounds = (currClustU32 < nClusters);
        if (inClusterBounds) {
          uint isSelected = selected_clusters[currClustU32];
          bool clusterSelected = (isSelected > 0u);
          if (clusterSelected) {
            foundLabel = currClust;
          } else {
            int parentCluster = cluster_parent_map[currClustU32];
            currentCluster = parentCluster;
          }
        }
      }
    }
    int finalLabel = foundLabel;
    labels[gid] = finalLabel;
  }
}
