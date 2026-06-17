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
__kernel void navatala_ml_compute_pairwise_affinity_f64(__global const double* X, __global const double* sigma, __global const uint* nPoints, __global const uint* nDims, __global double* P) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nPoints[0];
  uint d = nDims[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint i = (gid / n);
    uint j = (gid % n);
    double sigmaI = sigma[i];
    double twoSigmaSq = (as_double(0x4000000000000000ul) * (sigmaI * sigmaI));
    uint baseI = (i * d);
    uint baseJ = (j * d);
    double distSqAccum = as_double(0x0000000000000000ul);
    for (int k = 0; k < (int)(d); ++k) {
      uint kU32 = ((uint)(k));
      uint idxI = (baseI + kU32);
      uint idxJ = (baseJ + kU32);
      double xi_k = X[idxI];
      double xj_k = X[idxJ];
      double diff = (xi_k - xj_k);
      double diffSq = (diff * diff);
      double currentDistSq = distSqAccum;
      double newDistSq = (currentDistSq + diffSq);
      distSqAccum = newDistSq;
    }
    double distSq = distSqAccum;
    bool isDiag = (i == j);
    double negDistOverSigma = ((as_double(0x0000000000000000ul) - distSq) / twoSigmaSq);
    double pVal = exp(negDistOverSigma);
    if (isDiag) {
      P[gid] = as_double(0x0000000000000000ul);
    } else {
      P[gid] = pVal;
    }
  }
}
