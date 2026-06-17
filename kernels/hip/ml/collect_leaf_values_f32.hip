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
extern "C" __global__ void navatala_ml_collect_leaf_values_f32(const int* leaf_indices, const float* thresholds, const int* tree_offsets, const unsigned int* n_samples, const unsigned int* n_trees, float* leaf_values) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nTrees = n_trees[0];
  unsigned int totalWork = (nSamples * nTrees);
  bool inBounds = (gid < totalWork);
  if (inBounds) {
    unsigned int treeIdx = (gid % nTrees);
    int leafNodeIdx = leaf_indices[gid];
    unsigned int leafNodeIdxU32 = ((unsigned int)(leafNodeIdx));
    int treeOffset = tree_offsets[treeIdx];
    unsigned int treeOffsetU32 = ((unsigned int)(treeOffset));
    unsigned int absLeafIdx = (treeOffsetU32 + leafNodeIdxU32);
    float leafValue = thresholds[absLeafIdx];
    leaf_values[gid] = leafValue;
  }
}
