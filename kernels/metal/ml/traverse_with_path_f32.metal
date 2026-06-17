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

kernel void navatala_ml_traverse_with_path_f32(device const float* X [[buffer(0)]], device const int* feature_indices [[buffer(1)]], device const float* thresholds [[buffer(2)]], device const uint* left_children [[buffer(3)]], device const uint* right_children [[buffer(4)]], device const int* tree_offsets [[buffer(5)]], device const uint* n_samples [[buffer(6)]], device const uint* n_features [[buffer(7)]], device const uint* n_trees [[buffer(8)]], device const uint* max_depth [[buffer(9)]], device int* paths [[buffer(10)]], device int* path_lengths [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nTrees = n_trees[0];
  uint maxDepth = max_depth[0];
  uint totalWork = (nSamples * nTrees);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    uint sampleIdx = (gid / nTrees);
    uint treeIdx = (gid % nTrees);
    uint sampleBase = (sampleIdx * nFeatures);
    uint pathBase = (gid * maxDepth);
    int treeOffset = tree_offsets[treeIdx];
    uint treeOffsetU32 = ((uint)(treeOffset));
    for (int d = 0; d < (int)(maxDepth); ++d) {
      uint dU32 = ((uint)(d));
      uint pathIdx = (pathBase + dU32);
      paths[pathIdx] = -1;
    }
    uint currentNode = 0u;
    uint currentDepth = 0u;
    for (int iter = 0; iter < (int)(64u); ++iter) {
      uint nodeIdx = currentNode;
      uint depth = currentDepth;
      bool depthOk = (depth < maxDepth);
      if (depthOk) {
        uint pathIdx = (pathBase + depth);
        int nodeIdxI32 = ((int)(nodeIdx));
        paths[pathIdx] = nodeIdxI32;
        uint absNodeIdx = (treeOffsetU32 + nodeIdx);
        int featureIdx = feature_indices[absNodeIdx];
        float threshold = thresholds[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          int pathLen = ((int)((depth + 1u)));
          path_lengths[gid] = pathLen;
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
          uint newDepth = (depth + 1u);
          currentDepth = newDepth;
        }
      }
    }
  }
}
