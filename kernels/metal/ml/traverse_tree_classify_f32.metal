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

kernel void navatala_ml_traverse_tree_classify_f32(device const float* X [[buffer(0)]], device const int* feature_indices [[buffer(1)]], device const float* thresholds [[buffer(2)]], device const int* left_children [[buffer(3)]], device const int* right_children [[buffer(4)]], device const int* leaf_value_indices [[buffer(5)]], device const float* leaf_values [[buffer(6)]], device const uint* n_samples [[buffer(7)]], device const uint* n_features [[buffer(8)]], device const uint* n_classes [[buffer(9)]], device int* predictions [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    int currentNode = 0;
    for (int depth = 0; depth < (int)(64u); ++depth) {
      int nodeIdx = currentNode;
      uint nodeIdxU32 = ((uint)(nodeIdx));
      int featureIdx = feature_indices[nodeIdxU32];
      bool isLeaf = (featureIdx == -1);
      if (isLeaf) {
        int leafIdx = leaf_value_indices[nodeIdxU32];
        uint leafIdxU32 = ((uint)(leafIdx));
        uint leafBase = (leafIdxU32 * nClasses);
        int bestClass = 0;
        float bestProb = as_type<float>(0xbf800000u);
        for (int c = 0; c < (int)(nClasses); ++c) {
          uint cU32 = ((uint)(c));
          uint probIdx = (leafBase + cU32);
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
        uint featureIdxU32 = ((uint)(featureIdx));
        uint featureAddr = (sampleBase + featureIdxU32);
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
