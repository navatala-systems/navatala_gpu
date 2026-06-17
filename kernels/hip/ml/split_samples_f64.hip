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
extern "C" __global__ void navatala_ml_split_samples_f64(const double* X, const unsigned int* sample_indices, const unsigned int* n_node_samples, const unsigned int* n_features, const unsigned int* split_feature, const double* threshold, unsigned int* left_indices, unsigned int* right_indices, unsigned int* left_count, unsigned int* right_count) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_node_samples[0];
  unsigned int nFeatures = n_features[0];
  unsigned int splitFeat = split_feature[0];
  double thresh = threshold[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int sampleIdx = sample_indices[gid];
    unsigned int featureAddr = ((sampleIdx * nFeatures) + splitFeat);
    double featureVal = X[featureAddr];
    bool goLeft = (featureVal <= thresh);
    if (goLeft) {
      unsigned int pos = left_count[0];
      unsigned int newPos = (pos + 1u);
      left_count[0] = newPos;
      left_indices[pos] = sampleIdx;
    } else {
      unsigned int pos = right_count[0];
      unsigned int newPos = (pos + 1u);
      right_count[0] = newPos;
      right_indices[pos] = sampleIdx;
    }
  }
}
