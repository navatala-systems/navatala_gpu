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

__kernel void navatala_dataframe_welford_merge_f32(__global const uint* countA, __global const float* meanA, __global const float* M2A, __global const uint* countB, __global const float* meanB, __global const float* M2B, __global uint* count, __global float* mean, __global float* M2) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) == 0)) {
    uint nA = countA[0];
    float muA = meanA[0];
    float m2AVal = M2A[0];
    uint nB = countB[0];
    float muB = meanB[0];
    float m2BVal = M2B[0];
    uint nTotal = (nA + nB);
    float nAf = ((float)(nA));
    float nBf = ((float)(nB));
    float nTotalf = ((float)(nTotal));
    float delta = (muB - muA);
    float nTotalSafe = (((nTotal == (uint)(0u))) ? (as_float(0x3f800000u)) : (nTotalf));
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
