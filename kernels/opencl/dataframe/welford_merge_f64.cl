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
__kernel void navatala_dataframe_welford_merge_f64(__global const uint* countA, __global const double* meanA, __global const double* M2A, __global const uint* countB, __global const double* meanB, __global const double* M2B, __global uint* count, __global double* mean, __global double* M2) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) == 0) {
    uint nA = countA[0];
    double muA = meanA[0];
    double m2AVal = M2A[0];
    uint nB = countB[0];
    double muB = meanB[0];
    double m2BVal = M2B[0];
    uint nTotal = (nA + nB);
    double nAf = ((double)(nA));
    double nBf = ((double)(nB));
    double nTotalf = ((double)(nTotal));
    double delta = (muB - muA);
    double nTotalSafe = (((nTotal == (uint)(0u))) ? (as_double(0x3ff0000000000000ul)) : (nTotalf));
    double muTotal = (muA + (delta * (nBf / nTotalSafe)));
    double deltaSq = (delta * delta);
    double nAnB = (nAf * nBf);
    double correction = (deltaSq * (nAnB / nTotalSafe));
    double m2Total = ((m2AVal + m2BVal) + correction);
    count[0] = nTotal;
    mean[0] = muTotal;
    M2[0] = m2Total;
  }
}
