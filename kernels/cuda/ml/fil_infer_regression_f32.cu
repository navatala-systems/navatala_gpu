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
extern "C" __global__ void navatala_ml_fil_infer_regression_f32(const float* X, const int* feature_indices, const float* thresholds, const int* left_children, const int* right_children, const int* tree_offsets, const unsigned int* bias, const unsigned int* n_samples, const unsigned int* n_features, const unsigned int* n_trees, float* predictions) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nFeatures = n_features[0];
  unsigned int nTrees = n_trees[0];
  float biasVal = ((float)(bias[0]));
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int sampleBase = (gid * nFeatures);
    float sumOutput = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(nTrees); ++t) {
      unsigned int tU32 = ((unsigned int)(t));
      int treeOffset = tree_offsets[tU32];
      unsigned int treeOffsetU32 = ((unsigned int)(treeOffset));
      unsigned int nodeIdx = 0u;
      for (int depth = 0; depth < (int)(64u); ++depth) {
        unsigned int currNode = nodeIdx;
        unsigned int absNodeIdx = (treeOffsetU32 + currNode);
        int featureIdx = feature_indices[absNodeIdx];
        float threshold = thresholds[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          float currSum = sumOutput;
          float newSum = (currSum + threshold);
          sumOutput = newSum;
        } else {
          unsigned int featureIdxU32 = ((unsigned int)(featureIdx));
          unsigned int featureAddr = (sampleBase + featureIdxU32);
          float featureVal = X[featureAddr];
          bool goLeft = (featureVal <= threshold);
          if (goLeft) {
            int leftChild = left_children[absNodeIdx];
            unsigned int leftChildU32 = ((unsigned int)(leftChild));
            nodeIdx = leftChildU32;
          } else {
            int rightChild = right_children[absNodeIdx];
            unsigned int rightChildU32 = ((unsigned int)(rightChild));
            nodeIdx = rightChildU32;
          }
        }
      }
    }
    float finalSum = sumOutput;
    float nTreesF = ((float)(nTrees));
    float avgPred = ((finalSum / nTreesF) + biasVal);
    predictions[gid] = avgPred;
  }
}
