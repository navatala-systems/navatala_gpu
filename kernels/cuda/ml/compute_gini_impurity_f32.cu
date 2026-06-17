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
extern "C" __global__ void navatala_ml_compute_gini_impurity_f32(const int* labels, const unsigned int* sample_indices, const unsigned int* n_node_samples, const unsigned int* n_classes, float* gini, unsigned int* class_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int nSamples = n_node_samples[0];
  unsigned int nClasses = n_classes[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    unsigned int sampleIdx = sample_indices[gid];
    int label = labels[sampleIdx];
    unsigned int labelU32 = ((unsigned int)(label));
    unsigned int currCount = class_counts[labelU32];
    unsigned int newCount = (currCount + 1u);
    class_counts[labelU32] = newCount;
  }
  __syncthreads();
  bool isFirstThread = (gid == 0u);
  if (isFirstThread) {
    float sumPSquared = __uint_as_float(0x00000000u);
    float nSamplesF = ((float)(nSamples));
    for (int c = 0; c < (int)(nClasses); ++c) {
      unsigned int cU32 = ((unsigned int)(c));
      unsigned int count = class_counts[cU32];
      float countF = ((float)(count));
      float p = (countF / nSamplesF);
      float pSquared = (p * p);
      float currSum = sumPSquared;
      float newSum = (currSum + pSquared);
      sumPSquared = newSum;
    }
    float finalSum = sumPSquared;
    float giniVal = (__uint_as_float(0x3f800000u) - finalSum);
    gini[0] = giniVal;
  }
}
