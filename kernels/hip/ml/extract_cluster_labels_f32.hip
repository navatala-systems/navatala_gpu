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
extern "C" __global__ void navatala_ml_extract_cluster_labels_f32(const unsigned int* selected_clusters, const int* point_cluster_map, const int* cluster_parent_map, const unsigned int* n_samples, const unsigned int* n_clusters, int* labels) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nClusters = n_clusters[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    int initialCluster = point_cluster_map[gid];
    int currentCluster = initialCluster;
    int foundLabel = -1;
    for (int level = 0; level < (int)(64u); ++level) {
      int currClust = currentCluster;
      bool isValidCluster = (currClust >= 0);
      if (isValidCluster) {
        unsigned int currClustU32 = ((unsigned int)(currClust));
        bool inClusterBounds = (currClustU32 < nClusters);
        if (inClusterBounds) {
          unsigned int isSelected = selected_clusters[currClustU32];
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
