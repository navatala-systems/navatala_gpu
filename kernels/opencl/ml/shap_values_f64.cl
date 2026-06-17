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
__kernel void navatala_ml_shap_values_f64(__global const double* predictions, __global const double* weights, __global const double* coalitions, __global const uint* nSamples, __global const uint* nFeatures, __global double* shapValues) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamplesVal = nSamples[0];
  uint nFeaturesVal = nFeatures[0];
  uint featureIdx = gid;
  bool inBounds = (featureIdx < nFeaturesVal);
  if (inBounds) {
    double weightedSum = as_double(0x0000000000000000ul);
    double totalWeight = as_double(0x0000000000000000ul);
    for (int s = 0; s < (int)(nSamplesVal); ++s) {
      uint sU32 = ((uint)(s));
      uint coalIdx = ((sU32 * nFeaturesVal) + featureIdx);
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
    double eps = as_double(0x3cd203af9ee75616ul);
    double safeDenom = (finalTotalW + eps);
    double shapVal = (finalSum / safeDenom);
    shapValues[featureIdx] = shapVal;
  }
}
