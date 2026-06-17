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
extern "C" __global__ void navatala_dataframe_welford_merge_f32(const unsigned int* countA, const float* meanA, const float* M2A, const unsigned int* countB, const float* meanB, const float* M2B, unsigned int* count, float* mean, float* M2) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    unsigned int nA = countA[0];
    float muA = meanA[0];
    float m2AVal = M2A[0];
    unsigned int nB = countB[0];
    float muB = meanB[0];
    float m2BVal = M2B[0];
    unsigned int nTotal = (nA + nB);
    float nAf = ((float)(nA));
    float nBf = ((float)(nB));
    float nTotalf = ((float)(nTotal));
    float delta = (muB - muA);
    float nTotalSafe = (((nTotal == 0u)) ? (__uint_as_float(0x3f800000u)) : (nTotalf));
    float muTotal = (muA + (delta * (nBf / nTotalSafe)));
    float deltaSq = (delta * delta);
    float nAnB = (nAf * nBf);
    float correction = (deltaSq * (nAnB / nTotalSafe));
    float m2Total = ((m2AVal + m2BVal) + correction);
    count[0] = nTotal;
    mean[0] = muTotal;
    M2[0] = m2Total;
  }
}
