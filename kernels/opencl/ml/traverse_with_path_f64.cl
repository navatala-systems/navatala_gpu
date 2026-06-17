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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_ml_traverse_with_path_f64(__global const double* X, __global const int* feature_indices, __global const double* thresholds, __global const uint* left_children, __global const uint* right_children, __global const int* tree_offsets, __global const uint* n_samples, __global const uint* n_features, __global const uint* n_trees, __global const uint* max_depth, __global int* paths, __global int* path_lengths) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
    uint currentNode = (uint)(0u);
    uint currentDepth = (uint)(0u);
    for (int iter = 0; iter < (int)((uint)(64u)); ++iter) {
      uint nodeIdx = currentNode;
      uint depth = currentDepth;
      bool depthOk = (depth < maxDepth);
      if (depthOk) {
        uint pathIdx = (pathBase + depth);
        int nodeIdxI32 = ((int)(nodeIdx));
        paths[pathIdx] = nodeIdxI32;
        uint absNodeIdx = (treeOffsetU32 + nodeIdx);
        int featureIdx = feature_indices[absNodeIdx];
        double threshold = thresholds[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          int pathLen = ((int)((depth + (uint)(1u))));
          path_lengths[gid] = pathLen;
        } else {
          uint featureIdxU32 = ((uint)(featureIdx));
          uint featureAddr = (sampleBase + featureIdxU32);
          double featureVal = X[featureAddr];
          bool goLeft = (featureVal <= threshold);
          if (goLeft) {
            uint leftChild = left_children[absNodeIdx];
            currentNode = leftChild;
          } else {
            uint rightChild = right_children[absNodeIdx];
            currentNode = rightChild;
          }
          uint newDepth = (depth + (uint)(1u));
          currentDepth = newDepth;
        }
      }
    }
  }
}
