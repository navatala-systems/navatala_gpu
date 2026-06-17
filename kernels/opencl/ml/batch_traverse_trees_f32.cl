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

__kernel void navatala_ml_batch_traverse_trees_f32(__global const float* X, __global const int* feature_indices, __global const float* thresholds, __global const uint* left_children, __global const uint* right_children, __global const int* tree_offsets, __global const uint* n_samples, __global const uint* n_features, __global const uint* n_trees, __global int* leaf_indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nTrees = n_trees[0];
  uint totalWork = (nSamples * nTrees);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    uint sampleIdx = (gid / nTrees);
    uint treeIdx = (gid % nTrees);
    uint sampleBase = (sampleIdx * nFeatures);
    int treeOffset = tree_offsets[treeIdx];
    uint treeOffsetU32 = ((uint)(treeOffset));
    uint currentNode = (uint)(0u);
    for (int depth = 0; depth < (int)((uint)(64u)); ++depth) {
      uint nodeIdx = currentNode;
      uint absNodeIdx = (treeOffsetU32 + nodeIdx);
      int featureIdx = feature_indices[absNodeIdx];
      float threshold = thresholds[absNodeIdx];
      bool isLeaf = (featureIdx == -1);
      if (isLeaf) {
        int leafIdxI32 = ((int)(nodeIdx));
        leaf_indices[gid] = leafIdxI32;
      } else {
        uint featureIdxU32 = ((uint)(featureIdx));
        uint featureAddr = (sampleBase + featureIdxU32);
        float featureVal = X[featureAddr];
        bool goLeft = (featureVal <= threshold);
        if (goLeft) {
          uint leftChild = left_children[absNodeIdx];
          currentNode = leftChild;
        } else {
          uint rightChild = right_children[absNodeIdx];
          currentNode = rightChild;
        }
      }
    }
  }
}
