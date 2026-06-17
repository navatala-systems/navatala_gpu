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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_ml_batch_traverse_trees_f32(const float* X, const int* feature_indices, const float* thresholds, const unsigned int* left_children, const unsigned int* right_children, const int* tree_offsets, const unsigned int* n_samples, const unsigned int* n_features, const unsigned int* n_trees, int* leaf_indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nFeatures = n_features[0];
  unsigned int nTrees = n_trees[0];
  unsigned int totalWork = (nSamples * nTrees);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    unsigned int sampleIdx = (gid / nTrees);
    unsigned int treeIdx = (gid % nTrees);
    unsigned int sampleBase = (sampleIdx * nFeatures);
    int treeOffset = tree_offsets[treeIdx];
    unsigned int treeOffsetU32 = ((unsigned int)(treeOffset));
    unsigned int currentNode = 0u;
    for (int depth = 0; depth < (int)(64u); ++depth) {
      unsigned int nodeIdx = currentNode;
      unsigned int absNodeIdx = (treeOffsetU32 + nodeIdx);
      int featureIdx = feature_indices[absNodeIdx];
      float threshold = thresholds[absNodeIdx];
      bool isLeaf = (featureIdx == -1);
      if (isLeaf) {
        int leafIdxI32 = ((int)(nodeIdx));
        leaf_indices[gid] = leafIdxI32;
      } else {
        unsigned int featureIdxU32 = ((unsigned int)(featureIdx));
        unsigned int featureAddr = (sampleBase + featureIdxU32);
        float featureVal = X[featureAddr];
        bool goLeft = (featureVal <= threshold);
        if (goLeft) {
          unsigned int leftChild = left_children[absNodeIdx];
          currentNode = leftChild;
        } else {
          unsigned int rightChild = right_children[absNodeIdx];
          currentNode = rightChild;
        }
      }
    }
  }
}
