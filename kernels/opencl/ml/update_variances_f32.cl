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

__kernel void navatala_ml_update_variances_f32(__global const float* data, __global const float* means, __global const float* responsibilities, __global const uint* n, __global const uint* k, __global const uint* d, __global float* variances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numN = n[0];
  uint numK = k[0];
  uint numD = d[0];
  uint kd = (numK * numD);
  bool inBounds = (gid < kd);
  if (inBounds) {
    uint j = (gid / numD);
    uint l = (gid % numD);
    float mu_jl = means[gid];
    float weightedSqSumAccum = as_float(0x00000000u);
    float respSumAccum = as_float(0x00000000u);
    for (int i = 0; i < (int)(numN); ++i) {
      uint iU32 = ((uint)(i));
      uint respIdx = ((iU32 * numK) + j);
      float r_ij = responsibilities[respIdx];
      uint dataIdx = ((iU32 * numD) + l);
      float x_il = data[dataIdx];
      float diff = (x_il - mu_jl);
      float diffSq = (diff * diff);
      float weightedSq = (r_ij * diffSq);
      float currWeightedSqSum = weightedSqSumAccum;
      float newWeightedSqSum = (currWeightedSqSum + weightedSq);
      weightedSqSumAccum = newWeightedSqSum;
      float currRespSum = respSumAccum;
      float newRespSum = (currRespSum + r_ij);
      respSumAccum = newRespSum;
    }
    float weightedSqSum = weightedSqSumAccum;
    float respSum = respSumAccum;
    float rawVar = (weightedSqSum / respSum);
    float varVal = (rawVar + as_float(0x358637bdu));
    variances[gid] = varVal;
  }
}
