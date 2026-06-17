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
__kernel void navatala_ml_fil_infer_regression_f64(__global const double* X, __global const int* feature_indices, __global const double* thresholds, __global const int* left_children, __global const int* right_children, __global const int* tree_offsets, __global const uint* bias, __global const uint* n_samples, __global const uint* n_features, __global const uint* n_trees, __global double* predictions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  uint nTrees = n_trees[0];
  double biasVal = ((double)(bias[0]));
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    double sumOutput = as_double(0x0000000000000000ul);
    for (int t = 0; t < (int)(nTrees); ++t) {
      uint tU32 = ((uint)(t));
      int treeOffset = tree_offsets[tU32];
      uint treeOffsetU32 = ((uint)(treeOffset));
      uint nodeIdx = (uint)(0u);
      for (int depth = 0; depth < (int)((uint)(64u)); ++depth) {
        uint currNode = nodeIdx;
        uint absNodeIdx = (treeOffsetU32 + currNode);
        int featureIdx = feature_indices[absNodeIdx];
        double threshold = thresholds[absNodeIdx];
        bool isLeaf = (featureIdx == -1);
        if (isLeaf) {
          double currSum = sumOutput;
          double newSum = (currSum + threshold);
          sumOutput = newSum;
        } else {
          uint featureIdxU32 = ((uint)(featureIdx));
          uint featureAddr = (sampleBase + featureIdxU32);
          double featureVal = X[featureAddr];
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
    double finalSum = sumOutput;
    double nTreesF = ((double)(nTrees));
    double avgPred = ((finalSum / nTreesF) + biasVal);
    predictions[gid] = avgPred;
  }
}
