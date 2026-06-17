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

__kernel void navatala_ml_shap_values_f32(__global const float* predictions, __global const float* weights, __global const float* coalitions, __global const uint* nSamples, __global const uint* nFeatures, __global float* shapValues) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamplesVal = nSamples[0];
  uint nFeaturesVal = nFeatures[0];
  uint featureIdx = gid;
  bool inBounds = (featureIdx < nFeaturesVal);
  if (inBounds) {
    float weightedSum = as_float(0x00000000u);
    float totalWeight = as_float(0x00000000u);
    for (int s = 0; s < (int)(nSamplesVal); ++s) {
      uint sU32 = ((uint)(s));
      uint coalIdx = ((sU32 * nFeaturesVal) + featureIdx);
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
    float eps = as_float(0x2edbe6ffu);
    float safeDenom = (finalTotalW + eps);
    float shapVal = (finalSum / safeDenom);
    shapValues[featureIdx] = shapVal;
  }
}
