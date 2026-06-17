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

__kernel void navatala_ml_compute_objective_f32(__global const float* X, __global const float* beta, __global const float* y, __global const float* lambda, __global const uint* nSamples, __global const uint* nFeatures, __global float* objective) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdataMse[256];
  __local float sdataL1[256];
  uint n = nSamples[0];
  uint d = nFeatures[0];
  float lambdaVal = lambda[0];
  bool sampleInBounds = (gid < n);
  if (sampleInBounds) {
    float yVal = y[gid];
    uint rowOffset = (gid * d);
    float dotAccum = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint kU32 = ((uint)(k));
      uint xikIdx = (rowOffset + kU32);
      float xik = X[xikIdx];
      float betak = beta[kU32];
      float prod = (xik * betak);
      float currentDot = dotAccum;
      float newDot = (currentDot + prod);
      dotAccum = newDot;
    }
    float dotProduct = dotAccum;
    float residual = (yVal - dotProduct);
    float residSq = (residual * residual);
    sdataMse[lid] = residSq;
  } else {
    sdataMse[lid] = as_float(0x00000000u);
  }
  bool featureInBounds = (gid < d);
  if (featureInBounds) {
    float betaJ = beta[gid];
    float absBetaJ = fabs(betaJ);
    sdataL1[lid] = absBetaJ;
  } else {
    sdataL1[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint obj32RedStride = (uint)(128u);
  for (int obj32RedStep = 0; obj32RedStep < (int)(8); ++obj32RedStep) {
    uint obj32RedStrideVal = obj32RedStride;
    if ((lid < obj32RedStrideVal)) {
      float otherMse = sdataMse[(lid + obj32RedStrideVal)];
      float mineMse = sdataMse[lid];
      float sumMse = (mineMse + otherMse);
      sdataMse[lid] = sumMse;
      float otherL1 = sdataL1[(lid + obj32RedStrideVal)];
      float mineL1 = sdataL1[lid];
      float sumL1 = (mineL1 + otherL1);
      sdataL1[lid] = sumL1;
    }
    uint obj32RedStrideToHalve = obj32RedStride;
    uint obj32RedNextStride = (obj32RedStrideToHalve >> (uint)(1u));
    obj32RedStride = obj32RedNextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float totalMse = sdataMse[0];
    float totalL1 = sdataL1[0];
    float halfMse = (as_float(0x3f000000u) * totalMse);
    float lambdaL1 = (lambdaVal * totalL1);
    float obj = (halfMse + lambdaL1);
    objective[0] = obj;
  }
}
