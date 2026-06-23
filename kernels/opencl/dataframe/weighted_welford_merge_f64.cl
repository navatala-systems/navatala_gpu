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
__kernel void navatala_dataframe_weighted_welford_merge_f64(__global const double* weightSumA, __global const double* meanA, __global const double* M2A, __global const double* weightSumB, __global const double* meanB, __global const double* M2B, __global double* weightSum, __global double* mean, __global double* M2) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) == 0) {
    double wA = weightSumA[0];
    double muA = meanA[0];
    double m2AVal = M2A[0];
    double wB = weightSumB[0];
    double muB = meanB[0];
    double m2BVal = M2B[0];
    double wTotal = (wA + wB);
    double delta = (muB - muA);
    double wTotalSafe = (((wTotal == as_double(0x0000000000000000ul))) ? (as_double(0x3ff0000000000000ul)) : (wTotal));
    double muTotal = (muA + (delta * (wB / wTotalSafe)));
    double deltaSq = (delta * delta);
    double wAwB = (wA * wB);
    double correction = (deltaSq * (wAwB / wTotalSafe));
    double m2Total = ((m2AVal + m2BVal) + correction);
    weightSum[0] = wTotal;
    mean[0] = muTotal;
    M2[0] = m2Total;
  }
}
