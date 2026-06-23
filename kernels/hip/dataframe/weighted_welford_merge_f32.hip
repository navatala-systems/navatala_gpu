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
extern "C" __global__ void navatala_dataframe_weighted_welford_merge_f32(const float* weightSumA, const float* meanA, const float* M2A, const float* weightSumB, const float* meanB, const float* M2B, float* weightSum, float* mean, float* M2) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    float wA = weightSumA[0];
    float muA = meanA[0];
    float m2AVal = M2A[0];
    float wB = weightSumB[0];
    float muB = meanB[0];
    float m2BVal = M2B[0];
    float wTotal = (wA + wB);
    float delta = (muB - muA);
    float wTotalSafe = (((wTotal == __uint_as_float(0x00000000u))) ? (__uint_as_float(0x3f800000u)) : (wTotal));
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
