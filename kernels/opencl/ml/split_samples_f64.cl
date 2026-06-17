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
__kernel void navatala_ml_split_samples_f64(__global const double* X, __global const uint* sample_indices, __global const uint* n_node_samples, __global const uint* n_features, __global const uint* split_feature, __global const double* threshold, __global uint* left_indices, __global uint* right_indices, __global uint* left_count, __global uint* right_count) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_node_samples[0];
  uint nFeatures = n_features[0];
  uint splitFeat = split_feature[0];
  double thresh = threshold[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleIdx = sample_indices[gid];
    uint featureAddr = ((sampleIdx * nFeatures) + splitFeat);
    double featureVal = X[featureAddr];
    bool goLeft = (featureVal <= thresh);
    if (goLeft) {
      uint pos = left_count[0];
      uint newPos = (pos + (uint)(1u));
      left_count[0] = newPos;
      left_indices[pos] = sampleIdx;
    } else {
      uint pos = right_count[0];
      uint newPos = (pos + (uint)(1u));
      right_count[0] = newPos;
      right_indices[pos] = sampleIdx;
    }
  }
}
