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

__kernel void navatala_ml_coordinate_update_f32(__global const float* X, __global const float* residuals, __global const uint* j, __global const float* lambda, __global const uint* nSamples, __global const uint* nFeatures, __global float* newBetaJ) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdataDot[256];
  __local float sdataNorm[256];
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint jVal = j[0];
  float lambdaVal = lambda[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint xijIdx = ((gid * d) + jVal);
    float xij = X[xijIdx];
    float ri = residuals[gid];
    float dotProd = (xij * ri);
    float xijSq = (xij * xij);
    sdataDot[lid] = dotProd;
    sdataNorm[lid] = xijSq;
  } else {
    sdataDot[lid] = as_float(0x00000000u);
    sdataNorm[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint cu32RedStride = (uint)(128u);
  for (int cu32RedStep = 0; cu32RedStep < (int)(8); ++cu32RedStep) {
    uint cu32RedStrideVal = cu32RedStride;
    if ((lid < cu32RedStrideVal)) {
      float otherDot = sdataDot[(lid + cu32RedStrideVal)];
      float mineDot = sdataDot[lid];
      float sumDot = (mineDot + otherDot);
      sdataDot[lid] = sumDot;
      float otherNorm = sdataNorm[(lid + cu32RedStrideVal)];
      float mineNorm = sdataNorm[lid];
      float sumNorm = (mineNorm + otherNorm);
      sdataNorm[lid] = sumNorm;
    }
    uint cu32RedStrideToHalve = cu32RedStride;
    uint cu32RedNextStride = (cu32RedStrideToHalve >> (uint)(1u));
    cu32RedStride = cu32RedNextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float totalDot = sdataDot[0];
    float totalNorm = sdataNorm[0];
    float rho = (totalDot / totalNorm);
    float thresh = (lambdaVal / totalNorm);
    float absRho = fabs(rho);
    float diff = (absRho - thresh);
    float maxDiff = (((diff > as_float(0x00000000u))) ? (diff) : (as_float(0x00000000u)));
    bool positive = (rho > as_float(0x00000000u));
    bool negative = (rho < as_float(0x00000000u));
    float signVal = ((positive) ? (as_float(0x3f800000u)) : (((negative) ? (as_float(0xbf800000u)) : (as_float(0x00000000u)))));
    float newBeta = (signVal * maxDiff);
    newBetaJ[0] = newBeta;
  }
}
