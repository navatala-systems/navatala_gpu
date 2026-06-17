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

__kernel void navatala_ml_compute_gradient_f32(__global const float* P, __global const float* Q, __global const float* Y, __global const uint* nPoints, __global const uint* outputDims, __global float* grad) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint totalGradElems = (n * d);
  bool inBounds = (gid < totalGradElems);
  if (inBounds) {
    uint i = (gid / d);
    uint k = (gid % d);
    float yi_k = Y[gid];
    float gradSumAccum = as_float(0x00000000u);
    for (int j = 0; j < (int)(n); ++j) {
      uint jU32 = ((uint)(j));
      bool notDiag = (i != jU32);
      if (notDiag) {
        uint pIdx = ((i * n) + jU32);
        float pij = P[pIdx];
        float qij = Q[pIdx];
        uint yjIdx = ((jU32 * d) + k);
        float yj_k = Y[yjIdx];
        uint baseI = (i * d);
        uint baseJ = (jU32 * d);
        float distSqAccum = as_float(0x00000000u);
        for (int m = 0; m < (int)(d); ++m) {
          uint mU32 = ((uint)(m));
          uint idxIM = (baseI + mU32);
          uint idxJM = (baseJ + mU32);
          float yi_m = Y[idxIM];
          float yj_m = Y[idxJM];
          float diffM = (yi_m - yj_m);
          float diffSqM = (diffM * diffM);
          float currentDistSq = distSqAccum;
          float newDistSq = (currentDistSq + diffSqM);
          distSqAccum = newDistSq;
        }
        float distSq = distSqAccum;
        float invDist = (as_float(0x3f800000u) / (as_float(0x3f800000u) + distSq));
        float diffK = (yi_k - yj_k);
        float pMinusQ = (pij - qij);
        float term = (pMinusQ * (diffK * invDist));
        float currentGradSum = gradSumAccum;
        float newGradSum = (currentGradSum + term);
        gradSumAccum = newGradSum;
      }
    }
    float gradSum = gradSumAccum;
    float gradVal = (as_float(0x40800000u) * gradSum);
    grad[gid] = gradVal;
  }
}
