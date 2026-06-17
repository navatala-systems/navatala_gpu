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

__kernel void navatala_ml_compute_pairwise_affinity_f32(__global const float* X, __global const float* sigma, __global const uint* nPoints, __global const uint* nDims, __global float* P) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nPoints[0];
  uint d = nDims[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint i = (gid / n);
    uint j = (gid % n);
    float sigmaI = sigma[i];
    float twoSigmaSq = (as_float(0x40000000u) * (sigmaI * sigmaI));
    uint baseI = (i * d);
    uint baseJ = (j * d);
    float distSqAccum = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint kU32 = ((uint)(k));
      uint idxI = (baseI + kU32);
      uint idxJ = (baseJ + kU32);
      float xi_k = X[idxI];
      float xj_k = X[idxJ];
      float diff = (xi_k - xj_k);
      float diffSq = (diff * diff);
      float currentDistSq = distSqAccum;
      float newDistSq = (currentDistSq + diffSq);
      distSqAccum = newDistSq;
    }
    float distSq = distSqAccum;
    bool isDiag = (i == j);
    float negDistOverSigma = ((as_float(0x00000000u) - distSq) / twoSigmaSq);
    float pVal = exp(negDistOverSigma);
    if (isDiag) {
      P[gid] = as_float(0x00000000u);
    } else {
      P[gid] = pVal;
    }
  }
}
