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
__kernel void navatala_ml_compute_q_distribution_f64(__global const double* Y, __global const uint* nPoints, __global const uint* outputDims, __global double* Q, __global double* sumQ) {
  int gid0 = (int)get_global_id(0);
  __local double sdata[256];
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  double localQ = as_double(0x0000000000000000ul);
  if (inBounds) {
    uint i = (gid / n);
    uint j = (gid % n);
    uint baseI = (i * d);
    uint baseJ = (j * d);
    double distSqAccum = as_double(0x0000000000000000ul);
    for (int k = 0; k < (int)(d); ++k) {
      uint kU32 = ((uint)(k));
      uint idxI = (baseI + kU32);
      uint idxJ = (baseJ + kU32);
      double yi_k = Y[idxI];
      double yj_k = Y[idxJ];
      double diff = (yi_k - yj_k);
      double diffSq = (diff * diff);
      double currentDistSq = distSqAccum;
      double newDistSq = (currentDistSq + diffSq);
      distSqAccum = newDistSq;
    }
    double distSq = distSqAccum;
    bool isDiag = (i == j);
    double onePlusDistSq = (as_double(0x3ff0000000000000ul) + distSq);
    double qVal = (as_double(0x3ff0000000000000ul) / onePlusDistSq);
    if (isDiag) {
      Q[gid] = as_double(0x0000000000000000ul);
      sdata[lid] = as_double(0x0000000000000000ul);
    } else {
      Q[gid] = qVal;
      sdata[lid] = qVal;
    }
  } else {
    sdata[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint qF64ReductionStride = (uint)(128u);
  for (int qF64ReductionStep = 0; qF64ReductionStep < (int)(8); ++qF64ReductionStep) {
    uint qF64Stride = qF64ReductionStride;
    if (lid < qF64Stride) {
      double qF64Other = sdata[(lid + qF64Stride)];
      double qF64Mine = sdata[lid];
      double qF64Sum = (qF64Mine + qF64Other);
      sdata[lid] = qF64Sum;
    }
    uint qF64StrideToHalve = qF64ReductionStride;
    uint qF64NextStride = (qF64StrideToHalve >> (uint)(1u));
    qF64ReductionStride = qF64NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    double partialSum = sdata[0];
    double oldSum = sumQ[0];
    double newSum = (oldSum + partialSum);
    sumQ[0] = newSum;
  }
}
