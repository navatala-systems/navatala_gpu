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

__kernel void navatala_ml_knn_regress_f32(__global const uint* neighborIndices, __global const float* targetValues, __global const uint* q, __global const uint* k, __global float* predictions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint qVal = q[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    uint baseIdx = (gid * kVal);
    float sumAccum = as_float(0x00000000u);
    for (int j = 0; j < (int)(kVal); ++j) {
      uint jU32 = ((uint)(j));
      uint neighborOffset = (baseIdx + jU32);
      uint neighborIdx = neighborIndices[neighborOffset];
      float targetVal = targetValues[neighborIdx];
      float currentSum = sumAccum;
      float newSum = (currentSum + targetVal);
      sumAccum = newSum;
    }
    float finalSum = sumAccum;
    float kFloat = ((float)(kVal));
    float meanVal = (finalSum / kFloat);
    predictions[gid] = meanVal;
  }
}
