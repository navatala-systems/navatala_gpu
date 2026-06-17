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
extern "C" __global__ void navatala_ml_knn_regress_weighted_f64(const unsigned int* neighborIndices, const double* neighborDistances, const double* targetValues, const unsigned int* q, const unsigned int* k, double* predictions) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int qVal = q[0];
  unsigned int kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    unsigned int baseIdx = (gid * kVal);
    double weightedSumAccum = __longlong_as_double(0x0000000000000000ull);
    double weightSumAccum = __longlong_as_double(0x0000000000000000ull);
    double epsilon = __longlong_as_double(0x3cd203af9ee75616ull);
    for (int j = 0; j < (int)(kVal); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int neighborOffset = (baseIdx + jU32);
      unsigned int neighborIdx = neighborIndices[neighborOffset];
      double dist = neighborDistances[neighborOffset];
      double targetVal = targetValues[neighborIdx];
      double safeDist = (((dist < epsilon)) ? (epsilon) : (dist));
      double weight = (__longlong_as_double(0x3ff0000000000000ull) / safeDist);
      double currentWeightedSum = weightedSumAccum;
      double newWeightedSum = (currentWeightedSum + (targetVal * weight));
      weightedSumAccum = newWeightedSum;
      double currentWeightSum = weightSumAccum;
      double newWeightSum = (currentWeightSum + weight);
      weightSumAccum = newWeightSum;
    }
    double finalWeightedSum = weightedSumAccum;
    double finalWeightSum = weightSumAccum;
    double safeDenom = (((finalWeightSum == __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x3ff0000000000000ull)) : (finalWeightSum));
    double prediction = (finalWeightedSum / safeDenom);
    predictions[gid] = prediction;
  }
}
