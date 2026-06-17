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
__kernel void navatala_ml_traverse_tree_regress_f64(__global const double* X, __global const int* feature_indices, __global const double* thresholds, __global const int* left_children, __global const int* right_children, __global const int* leaf_value_indices, __global const double* leaf_values, __global const uint* n_samples, __global const uint* n_features, __global double* predictions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nFeatures = n_features[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    int currentNode = 0;
    for (int depth = 0; depth < (int)((uint)(64u)); ++depth) {
      int nodeIdx = currentNode;
      uint nodeIdxU32 = ((uint)(nodeIdx));
      int featureIdx = feature_indices[nodeIdxU32];
      bool isLeaf = (featureIdx == -1);
      if (isLeaf) {
        int leafIdx = leaf_value_indices[nodeIdxU32];
        uint leafIdxU32 = ((uint)(leafIdx));
        double leafVal = leaf_values[leafIdxU32];
        predictions[gid] = leafVal;
      } else {
        double threshold = thresholds[nodeIdxU32];
        uint featureIdxU32 = ((uint)(featureIdx));
        uint featureAddr = (sampleBase + featureIdxU32);
        double featureVal = X[featureAddr];
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
