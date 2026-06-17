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
__kernel void navatala_ml_compute_objective_f64(__global const double* X, __global const double* beta, __global const double* y, __global const double* lambda, __global const uint* nSamples, __global const uint* nFeatures, __global double* objective) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdataMse[256];
  __local double sdataL1[256];
  uint n = nSamples[0];
  uint d = nFeatures[0];
  double lambdaVal = lambda[0];
  bool sampleInBounds = (gid < n);
  if (sampleInBounds) {
    double yVal = y[gid];
    uint rowOffset = (gid * d);
    double dotAccum = as_double(0x0000000000000000ul);
    for (int k = 0; k < (int)(d); ++k) {
      uint kU32 = ((uint)(k));
      uint xikIdx = (rowOffset + kU32);
      double xik = X[xikIdx];
      double betak = beta[kU32];
      double prod = (xik * betak);
      double currentDot = dotAccum;
      double newDot = (currentDot + prod);
      dotAccum = newDot;
    }
    double dotProduct = dotAccum;
    double residual = (yVal - dotProduct);
    double residSq = (residual * residual);
    sdataMse[lid] = residSq;
  } else {
    sdataMse[lid] = as_double(0x0000000000000000ul);
  }
  bool featureInBounds = (gid < d);
  if (featureInBounds) {
    double betaJ = beta[gid];
    double absBetaJ = fabs(betaJ);
    sdataL1[lid] = absBetaJ;
  } else {
    sdataL1[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint obj64RedStride = (uint)(128u);
  for (int obj64RedStep = 0; obj64RedStep < (int)(8); ++obj64RedStep) {
    uint obj64RedStrideVal = obj64RedStride;
    if ((lid < obj64RedStrideVal)) {
      double otherMse = sdataMse[(lid + obj64RedStrideVal)];
      double mineMse = sdataMse[lid];
      double sumMse = (mineMse + otherMse);
      sdataMse[lid] = sumMse;
      double otherL1 = sdataL1[(lid + obj64RedStrideVal)];
      double mineL1 = sdataL1[lid];
      double sumL1 = (mineL1 + otherL1);
      sdataL1[lid] = sumL1;
    }
    uint obj64RedStrideToHalve = obj64RedStride;
    uint obj64RedNextStride = (obj64RedStrideToHalve >> (uint)(1u));
    obj64RedStride = obj64RedNextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    double totalMse = sdataMse[0];
    double totalL1 = sdataL1[0];
    double halfMse = (as_double(0x3fe0000000000000ul) * totalMse);
    double lambdaL1 = (lambdaVal * totalL1);
    double obj = (halfMse + lambdaL1);
    objective[0] = obj;
  }
}
