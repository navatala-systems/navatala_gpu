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
__kernel void navatala_ml_compute_k_l_divergence_f64(__global const double* P, __global const double* Q, __global const uint* nPoints, __global double* kl_divergence) {
  int gid0 = (int)get_global_id(0);
  __local double sdata[256];
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint n = nPoints[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    double pij = P[gid];
    double qij = Q[gid];
    double minVal = as_double(0x3e7ad7f29abcaf48ul);
    bool pSafe = (pij > minVal);
    bool qSafe = (qij > minVal);
    bool bothSafe = (pSafe && qSafe);
    if (bothSafe) {
      double ratio = (pij / qij);
      double logRatio = log(ratio);
      double klTerm = (pij * logRatio);
      sdata[lid] = klTerm;
    } else {
      sdata[lid] = as_double(0x0000000000000000ul);
    }
  } else {
    sdata[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint klF64ReductionStride = (uint)(128u);
  for (int klF64ReductionStep = 0; klF64ReductionStep < (int)(8); ++klF64ReductionStep) {
    uint klF64Stride = klF64ReductionStride;
    if ((lid < klF64Stride)) {
      double klF64Other = sdata[(lid + klF64Stride)];
      double klF64Mine = sdata[lid];
      double klF64Sum = (klF64Mine + klF64Other);
      sdata[lid] = klF64Sum;
    }
    uint klF64StrideToHalve = klF64ReductionStride;
    uint klF64NextStride = (klF64StrideToHalve >> (uint)(1u));
    klF64ReductionStride = klF64NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    double klVal = sdata[0];
    kl_divergence[0] = klVal;
  }
}
