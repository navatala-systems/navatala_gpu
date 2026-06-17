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
extern "C" __global__ void navatala_ml_shap_values_f32(const float* predictions, const float* weights, const float* coalitions, const unsigned int* nSamples, const unsigned int* nFeatures, float* shapValues) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamplesVal = nSamples[0];
  unsigned int nFeaturesVal = nFeatures[0];
  unsigned int featureIdx = gid;
  bool inBounds = (featureIdx < nFeaturesVal);
  if (inBounds) {
    float weightedSum = __uint_as_float(0x00000000u);
    float totalWeight = __uint_as_float(0x00000000u);
    for (int s = 0; s < (int)(nSamplesVal); ++s) {
      unsigned int sU32 = ((unsigned int)(s));
      unsigned int coalIdx = ((sU32 * nFeaturesVal) + featureIdx);
      float coalMembership = coalitions[coalIdx];
      float w = weights[sU32];
      float pred = predictions[sU32];
      float contribution = (w * (coalMembership * pred));
      float currentSum = weightedSum;
      float newSum = (currentSum + contribution);
      weightedSum = newSum;
      float weightContrib = (w * coalMembership);
      float currentTotalW = totalWeight;
      float newTotalW = (currentTotalW + weightContrib);
      totalWeight = newTotalW;
    }
    float finalSum = weightedSum;
    float finalTotalW = totalWeight;
    float eps = __uint_as_float(0x2edbe6ffu);
    float safeDenom = (finalTotalW + eps);
    float shapVal = (finalSum / safeDenom);
    shapValues[featureIdx] = shapVal;
  }
}
