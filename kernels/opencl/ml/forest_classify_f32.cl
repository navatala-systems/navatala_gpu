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

__kernel void navatala_ml_forest_classify_f32(__global const float* X, __global const int* feature_indices, __global const float* thresholds, __global const int* left_children, __global const int* right_children, __global const int* leaf_value_indices, __global const float* forest_leaves, __global const int* tree_offsets, __global const int* leaf_offsets, __global const uint* n_samples, __global const uint* n_features, __global const uint* n_trees, __global const uint* n_classes, __global int* predictions, __global uint* vote_counts) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nTrees = n_trees[0];
  uint nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    uint voteBase = (gid * nClasses);
    for (int c = 0; c < (int)(nClasses); ++c) {
      uint cU32 = ((uint)(c));
      uint voteIdx = (voteBase + cU32);
      vote_counts[voteIdx] = (uint)(0u);
    }
    for (int t = 0; t < (int)(nTrees); ++t) {
      uint tU32 = ((uint)(t));
      int treeNodeOffset = tree_offsets[tU32];
      uint treeNodeOffsetU32 = ((uint)(treeNodeOffset));
      int treeLeafOffset = leaf_offsets[tU32];
      uint treeLeafOffsetU32 = ((uint)(treeLeafOffset));
      int currentNode = 0;
      for (int depth = 0; depth < (int)((uint)(64u)); ++depth) {
        int nodeIdx = currentNode;
        uint nodeIdxU32 = ((uint)(nodeIdx));
        uint absNodeIdx = (treeNodeOffsetU32 + nodeIdxU32);
        int featureIdx = feature_indices[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          int leafIdxI32 = leaf_value_indices[absNodeIdx];
          uint leafIdx = ((uint)(leafIdxI32));
          uint absLeafBase = (treeLeafOffsetU32 + (leafIdx * nClasses));
          int bestClass = 0;
          float bestProb = as_float(0xbf800000u);
          for (int classIdx = 0; classIdx < (int)(nClasses); ++classIdx) {
            uint classIdxU32 = ((uint)(classIdx));
            uint probIdx = (absLeafBase + classIdxU32);
            float prob = forest_leaves[probIdx];
            float currBest = bestProb;
            if ((prob > currBest)) {
              bestClass = classIdx;
              bestProb = prob;
            }
          }
          int predictedClass = bestClass;
          uint predictedClassU32 = ((uint)(predictedClass));
          uint voteIdx = (voteBase + predictedClassU32);
          uint currVotes = vote_counts[voteIdx];
          uint newVotes = (currVotes + (uint)(1u));
          vote_counts[voteIdx] = newVotes;
        } else {
          float threshold = thresholds[absNodeIdx];
          uint featureIdxU32 = ((uint)(featureIdx));
          uint featureAddr = (sampleBase + featureIdxU32);
          float featureVal = X[featureAddr];
          bool goLeft = (featureVal <= threshold);
          if (goLeft) {
            int leftChild = left_children[absNodeIdx];
            currentNode = leftChild;
          } else {
            int rightChild = right_children[absNodeIdx];
            currentNode = rightChild;
          }
        }
      }
    }
    int finalClass = 0;
    uint maxVotes = (uint)(0u);
    for (int c = 0; c < (int)(nClasses); ++c) {
      uint cU32 = ((uint)(c));
      uint voteIdx = (voteBase + cU32);
      uint votes = vote_counts[voteIdx];
      uint currMax = maxVotes;
      if ((votes > currMax)) {
        finalClass = c;
        maxVotes = votes;
      }
    }
    int result = finalClass;
    predictions[gid] = result;
  }
}
