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

__kernel void navatala_ml_extract_cluster_labels_f32(__global const uint* selected_clusters, __global const int* point_cluster_map, __global const int* cluster_parent_map, __global const uint* n_samples, __global const uint* n_clusters, __global int* labels) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nClusters = n_clusters[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    int initialCluster = point_cluster_map[gid];
    int currentCluster = initialCluster;
    int foundLabel = -1;
    for (int level = 0; level < (int)((uint)(64u)); ++level) {
      int currClust = currentCluster;
      bool isValidCluster = (currClust >= 0);
      if (isValidCluster) {
        uint currClustU32 = ((uint)(currClust));
        bool inClusterBounds = (currClustU32 < nClusters);
        if (inClusterBounds) {
          uint isSelected = selected_clusters[currClustU32];
          bool clusterSelected = (isSelected > (uint)(0u));
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
