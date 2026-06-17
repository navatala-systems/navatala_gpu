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

__kernel void navatala_ml_knn_regress_weighted_f32(__global const uint* neighborIndices, __global const float* neighborDistances, __global const float* targetValues, __global const uint* q, __global const uint* k, __global float* predictions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint qVal = q[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    uint baseIdx = (gid * kVal);
    float weightedSumAccum = as_float(0x00000000u);
    float weightSumAccum = as_float(0x00000000u);
    float epsilon = as_float(0x2edbe6ffu);
    for (int j = 0; j < (int)(kVal); ++j) {
      uint jU32 = ((uint)(j));
      uint neighborOffset = (baseIdx + jU32);
      uint neighborIdx = neighborIndices[neighborOffset];
      float dist = neighborDistances[neighborOffset];
      float targetVal = targetValues[neighborIdx];
      float safeDist = (((dist < epsilon)) ? (epsilon) : (dist));
      float weight = (as_float(0x3f800000u) / safeDist);
      float currentWeightedSum = weightedSumAccum;
      float newWeightedSum = (currentWeightedSum + (targetVal * weight));
      weightedSumAccum = newWeightedSum;
      float currentWeightSum = weightSumAccum;
      float newWeightSum = (currentWeightSum + weight);
      weightSumAccum = newWeightSum;
    }
    float finalWeightedSum = weightedSumAccum;
    float finalWeightSum = weightSumAccum;
    float safeDenom = (((finalWeightSum == as_float(0x00000000u))) ? (as_float(0x3f800000u)) : (finalWeightSum));
    float prediction = (finalWeightedSum / safeDenom);
    predictions[gid] = prediction;
  }
}
