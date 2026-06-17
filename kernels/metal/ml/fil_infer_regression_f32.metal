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

kernel void navatala_ml_fil_infer_regression_f32(device const float* X [[buffer(0)]], device const int* feature_indices [[buffer(1)]], device const float* thresholds [[buffer(2)]], device const int* left_children [[buffer(3)]], device const int* right_children [[buffer(4)]], device const int* tree_offsets [[buffer(5)]], device const uint* bias [[buffer(6)]], device const uint* n_samples [[buffer(7)]], device const uint* n_features [[buffer(8)]], device const uint* n_trees [[buffer(9)]], device float* predictions [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nTrees = n_trees[0];
  float biasVal = ((float)(bias[0]));
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    float sumOutput = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(nTrees); ++t) {
      uint tU32 = ((uint)(t));
      int treeOffset = tree_offsets[tU32];
      uint treeOffsetU32 = ((uint)(treeOffset));
      uint nodeIdx = 0u;
      for (int depth = 0; depth < (int)(64u); ++depth) {
        uint currNode = nodeIdx;
        uint absNodeIdx = (treeOffsetU32 + currNode);
        int featureIdx = feature_indices[absNodeIdx];
        float threshold = thresholds[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          float currSum = sumOutput;
          float newSum = (currSum + threshold);
          sumOutput = newSum;
        } else {
          uint featureIdxU32 = ((uint)(featureIdx));
          uint featureAddr = (sampleBase + featureIdxU32);
          float featureVal = X[featureAddr];
          bool goLeft = (featureVal <= threshold);
          if (goLeft) {
            int leftChild = left_children[absNodeIdx];
            uint leftChildU32 = ((uint)(leftChild));
            nodeIdx = leftChildU32;
          } else {
            int rightChild = right_children[absNodeIdx];
            uint rightChildU32 = ((uint)(rightChild));
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
