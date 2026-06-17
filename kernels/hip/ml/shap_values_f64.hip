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
extern "C" __global__ void navatala_ml_shap_values_f64(const double* predictions, const double* weights, const double* coalitions, const unsigned int* nSamples, const unsigned int* nFeatures, double* shapValues) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamplesVal = nSamples[0];
  unsigned int nFeaturesVal = nFeatures[0];
  unsigned int featureIdx = gid;
  bool inBounds = (featureIdx < nFeaturesVal);
  if (inBounds) {
    double weightedSum = __longlong_as_double(0x0000000000000000ull);
    double totalWeight = __longlong_as_double(0x0000000000000000ull);
    for (int s = 0; s < (int)(nSamplesVal); ++s) {
      unsigned int sU32 = ((unsigned int)(s));
      unsigned int coalIdx = ((sU32 * nFeaturesVal) + featureIdx);
      double coalMembership = coalitions[coalIdx];
      double w = weights[sU32];
      double pred = predictions[sU32];
      double contribution = (w * (coalMembership * pred));
      double currentSum = weightedSum;
      double newSum = (currentSum + contribution);
      weightedSum = newSum;
      double weightContrib = (w * coalMembership);
      double currentTotalW = totalWeight;
      double newTotalW = (currentTotalW + weightContrib);
      totalWeight = newTotalW;
    }
    double finalSum = weightedSum;
    double finalTotalW = totalWeight;
    double eps = __longlong_as_double(0x3cd203af9ee75616ull);
    double safeDenom = (finalTotalW + eps);
    double shapVal = (finalSum / safeDenom);
    shapValues[featureIdx] = shapVal;
  }
}
