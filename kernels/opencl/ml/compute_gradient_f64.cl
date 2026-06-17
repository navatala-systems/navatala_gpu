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
__kernel void navatala_ml_compute_gradient_f64(__global const double* P, __global const double* Q, __global const double* Y, __global const uint* nPoints, __global const uint* outputDims, __global double* grad) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint totalGradElems = (n * d);
  bool inBounds = (gid < totalGradElems);
  if (inBounds) {
    uint i = (gid / d);
    uint k = (gid % d);
    double yi_k = Y[gid];
    double gradSumAccum = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)(n); ++j) {
      uint jU32 = ((uint)(j));
      bool notDiag = (i != jU32);
      if (notDiag) {
        uint pIdx = ((i * n) + jU32);
        double pij = P[pIdx];
        double qij = Q[pIdx];
        uint yjIdx = ((jU32 * d) + k);
        double yj_k = Y[yjIdx];
        uint baseI = (i * d);
        uint baseJ = (jU32 * d);
        double distSqAccum = as_double(0x0000000000000000ul);
        for (int m = 0; m < (int)(d); ++m) {
          uint mU32 = ((uint)(m));
          uint idxIM = (baseI + mU32);
          uint idxJM = (baseJ + mU32);
          double yi_m = Y[idxIM];
          double yj_m = Y[idxJM];
          double diffM = (yi_m - yj_m);
          double diffSqM = (diffM * diffM);
          double currentDistSq = distSqAccum;
          double newDistSq = (currentDistSq + diffSqM);
          distSqAccum = newDistSq;
        }
        double distSq = distSqAccum;
        double invDist = (as_double(0x3ff0000000000000ul) / (as_double(0x3ff0000000000000ul) + distSq));
        double diffK = (yi_k - yj_k);
        double pMinusQ = (pij - qij);
        double term = (pMinusQ * (diffK * invDist));
        double currentGradSum = gradSumAccum;
        double newGradSum = (currentGradSum + term);
        gradSumAccum = newGradSum;
      }
    }
    double gradSum = gradSumAccum;
    double gradVal = (as_double(0x4010000000000000ul) * gradSum);
    grad[gid] = gradVal;
  }
}
