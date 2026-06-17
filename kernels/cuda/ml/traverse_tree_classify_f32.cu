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
extern "C" __global__ void navatala_ml_traverse_tree_classify_f32(const float* X, const int* feature_indices, const float* thresholds, const int* left_children, const int* right_children, const int* leaf_value_indices, const float* leaf_values, const unsigned int* n_samples, const unsigned int* n_features, const unsigned int* n_classes, int* predictions) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nFeatures = n_features[0];
  unsigned int nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int sampleBase = (gid * nFeatures);
    int currentNode = 0;
    for (int depth = 0; depth < (int)(64u); ++depth) {
      int nodeIdx = currentNode;
      unsigned int nodeIdxU32 = ((unsigned int)(nodeIdx));
      int featureIdx = feature_indices[nodeIdxU32];
      bool isLeaf = (featureIdx == -1);
      if (isLeaf) {
        int leafIdx = leaf_value_indices[nodeIdxU32];
        unsigned int leafIdxU32 = ((unsigned int)(leafIdx));
        unsigned int leafBase = (leafIdxU32 * nClasses);
        int bestClass = 0;
        float bestProb = __uint_as_float(0xbf800000u);
        for (int c = 0; c < (int)(nClasses); ++c) {
          unsigned int cU32 = ((unsigned int)(c));
          unsigned int probIdx = (leafBase + cU32);
          float prob = leaf_values[probIdx];
          float currBest = bestProb;
          if ((prob > currBest)) {
            bestClass = c;
            bestProb = prob;
          }
        }
        int finalClass = bestClass;
        predictions[gid] = finalClass;
      } else {
        float threshold = thresholds[nodeIdxU32];
        unsigned int featureIdxU32 = ((unsigned int)(featureIdx));
        unsigned int featureAddr = (sampleBase + featureIdxU32);
        float featureVal = X[featureAddr];
        bool goLeft = (featureVal <= threshold);
        if (goLeft) {
          int leftChild = left_children[nodeIdxU32];
          currentNode = leftChild;
        } else {
          int rightChild = right_children[nodeIdxU32];
          currentNode = rightChild;
        }
      }
    }
  }
}
