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
extern "C" __global__ void navatala_ml_forest_regress_f32(const float* X, const int* feature_indices, const float* thresholds, const int* left_children, const int* right_children, const int* leaf_value_indices, const float* forest_leaves, const int* tree_offsets, const int* leaf_offsets, const unsigned int* n_samples, const unsigned int* n_features, const unsigned int* n_trees, float* predictions) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nFeatures = n_features[0];
  unsigned int nTrees = n_trees[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int sampleBase = (gid * nFeatures);
    float sumPreds = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(nTrees); ++t) {
      unsigned int tU32 = ((unsigned int)(t));
      int treeNodeOffset = tree_offsets[tU32];
      unsigned int treeNodeOffsetU32 = ((unsigned int)(treeNodeOffset));
      int treeLeafOffset = leaf_offsets[tU32];
      unsigned int treeLeafOffsetU32 = ((unsigned int)(treeLeafOffset));
      int currentNode = 0;
      for (int depth = 0; depth < (int)(64u); ++depth) {
        int nodeIdx = currentNode;
        unsigned int nodeIdxU32 = ((unsigned int)(nodeIdx));
        unsigned int absNodeIdx = (treeNodeOffsetU32 + nodeIdxU32);
        int featureIdx = feature_indices[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          int leafIdxI32 = leaf_value_indices[absNodeIdx];
          unsigned int leafIdx = ((unsigned int)(leafIdxI32));
          unsigned int absLeafIdx = (treeLeafOffsetU32 + leafIdx);
          float leafVal = forest_leaves[absLeafIdx];
          float currSum = sumPreds;
          float newSum = (currSum + leafVal);
          sumPreds = newSum;
        } else {
          float threshold = thresholds[absNodeIdx];
          unsigned int featureIdxU32 = ((unsigned int)(featureIdx));
          unsigned int featureAddr = (sampleBase + featureIdxU32);
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
    float finalSum = sumPreds;
    float nTreesF = ((float)(nTrees));
    float avgPred = (finalSum / nTreesF);
    predictions[gid] = avgPred;
  }
}
