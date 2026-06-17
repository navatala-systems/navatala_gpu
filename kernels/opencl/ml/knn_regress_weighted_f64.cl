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
__kernel void navatala_ml_knn_regress_weighted_f64(__global const uint* neighborIndices, __global const double* neighborDistances, __global const double* targetValues, __global const uint* q, __global const uint* k, __global double* predictions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint qVal = q[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    uint baseIdx = (gid * kVal);
    double weightedSumAccum = as_double(0x0000000000000000ul);
    double weightSumAccum = as_double(0x0000000000000000ul);
    double epsilon = as_double(0x3cd203af9ee75616ul);
    for (int j = 0; j < (int)(kVal); ++j) {
      uint jU32 = ((uint)(j));
      uint neighborOffset = (baseIdx + jU32);
      uint neighborIdx = neighborIndices[neighborOffset];
      double dist = neighborDistances[neighborOffset];
      double targetVal = targetValues[neighborIdx];
      double safeDist = (((dist < epsilon)) ? (epsilon) : (dist));
      double weight = (as_double(0x3ff0000000000000ul) / safeDist);
      double currentWeightedSum = weightedSumAccum;
      double newWeightedSum = (currentWeightedSum + (targetVal * weight));
      weightedSumAccum = newWeightedSum;
      double currentWeightSum = weightSumAccum;
      double newWeightSum = (currentWeightSum + weight);
      weightSumAccum = newWeightSum;
    }
    double finalWeightedSum = weightedSumAccum;
    double finalWeightSum = weightSumAccum;
    double safeDenom = (((finalWeightSum == as_double(0x0000000000000000ul))) ? (as_double(0x3ff0000000000000ul)) : (finalWeightSum));
    double prediction = (finalWeightedSum / safeDenom);
    predictions[gid] = prediction;
  }
}
