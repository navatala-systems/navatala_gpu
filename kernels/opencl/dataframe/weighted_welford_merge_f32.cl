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

__kernel void navatala_dataframe_weighted_welford_merge_f32(__global const float* weightSumA, __global const float* meanA, __global const float* M2A, __global const float* weightSumB, __global const float* meanB, __global const float* M2B, __global float* weightSum, __global float* mean, __global float* M2) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) == 0)) {
    float wA = weightSumA[0];
    float muA = meanA[0];
    float m2AVal = M2A[0];
    float wB = weightSumB[0];
    float muB = meanB[0];
    float m2BVal = M2B[0];
    float wTotal = (wA + wB);
    float delta = (muB - muA);
    float wTotalSafe = (((wTotal == as_float(0x00000000u))) ? (as_float(0x3f800000u)) : (wTotal));
    float muTotal = (muA + (delta * (wB / wTotalSafe)));
    float deltaSq = (delta * delta);
    float wAwB = (wA * wB);
    float correction = (deltaSq * (wAwB / wTotalSafe));
    float m2Total = ((m2AVal + m2BVal) + correction);
    weightSum[0] = wTotal;
    mean[0] = muTotal;
    M2[0] = m2Total;
  }
}
