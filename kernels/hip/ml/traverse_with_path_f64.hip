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
extern "C" __global__ void navatala_ml_traverse_with_path_f64(const double* X, const int* feature_indices, const double* thresholds, const unsigned int* left_children, const unsigned int* right_children, const int* tree_offsets, const unsigned int* n_samples, const unsigned int* n_features, const unsigned int* n_trees, const unsigned int* max_depth, int* paths, int* path_lengths) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nFeatures = n_features[0];
  unsigned int nTrees = n_trees[0];
  unsigned int maxDepth = max_depth[0];
  unsigned int totalWork = (nSamples * nTrees);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    unsigned int sampleIdx = (gid / nTrees);
    unsigned int treeIdx = (gid % nTrees);
    unsigned int sampleBase = (sampleIdx * nFeatures);
    unsigned int pathBase = (gid * maxDepth);
    int treeOffset = tree_offsets[treeIdx];
    unsigned int treeOffsetU32 = ((unsigned int)(treeOffset));
    for (int d = 0; d < (int)(maxDepth); ++d) {
      unsigned int dU32 = ((unsigned int)(d));
      unsigned int pathIdx = (pathBase + dU32);
      paths[pathIdx] = -1;
    }
    unsigned int currentNode = 0u;
    unsigned int currentDepth = 0u;
    for (int iter = 0; iter < (int)(64u); ++iter) {
      unsigned int nodeIdx = currentNode;
      unsigned int depth = currentDepth;
      bool depthOk = (depth < maxDepth);
      if (depthOk) {
        unsigned int pathIdx = (pathBase + depth);
        int nodeIdxI32 = ((int)(nodeIdx));
        paths[pathIdx] = nodeIdxI32;
        unsigned int absNodeIdx = (treeOffsetU32 + nodeIdx);
        int featureIdx = feature_indices[absNodeIdx];
        double threshold = thresholds[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          int pathLen = ((int)((depth + 1u)));
          path_lengths[gid] = pathLen;
        } else {
          unsigned int featureIdxU32 = ((unsigned int)(featureIdx));
          unsigned int featureAddr = (sampleBase + featureIdxU32);
          double featureVal = X[featureAddr];
          bool goLeft = (featureVal <= threshold);
          if (goLeft) {
            unsigned int leftChild = left_children[absNodeIdx];
            currentNode = leftChild;
          } else {
            unsigned int rightChild = right_children[absNodeIdx];
            currentNode = rightChild;
          }
          unsigned int newDepth = (depth + 1u);
          currentDepth = newDepth;
        }
      }
    }
  }
}
