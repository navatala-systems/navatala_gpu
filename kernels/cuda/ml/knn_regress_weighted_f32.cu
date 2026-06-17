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
extern "C" __global__ void navatala_ml_knn_regress_weighted_f32(const unsigned int* neighborIndices, const float* neighborDistances, const float* targetValues, const unsigned int* q, const unsigned int* k, float* predictions) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int qVal = q[0];
  unsigned int kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    unsigned int baseIdx = (gid * kVal);
    float weightedSumAccum = __uint_as_float(0x00000000u);
    float weightSumAccum = __uint_as_float(0x00000000u);
    float epsilon = __uint_as_float(0x2edbe6ffu);
    for (int j = 0; j < (int)(kVal); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int neighborOffset = (baseIdx + jU32);
      unsigned int neighborIdx = neighborIndices[neighborOffset];
      float dist = neighborDistances[neighborOffset];
      float targetVal = targetValues[neighborIdx];
      float safeDist = (((dist < epsilon)) ? (epsilon) : (dist));
      float weight = (__uint_as_float(0x3f800000u) / safeDist);
      float currentWeightedSum = weightedSumAccum;
      float newWeightedSum = (currentWeightedSum + (targetVal * weight));
      weightedSumAccum = newWeightedSum;
      float currentWeightSum = weightSumAccum;
      float newWeightSum = (currentWeightSum + weight);
      weightSumAccum = newWeightSum;
    }
    float finalWeightedSum = weightedSumAccum;
    float finalWeightSum = weightSumAccum;
    float safeDenom = (((finalWeightSum == __uint_as_float(0x00000000u))) ? (__uint_as_float(0x3f800000u)) : (finalWeightSum));
    float prediction = (finalWeightedSum / safeDenom);
    predictions[gid] = prediction;
  }
}
