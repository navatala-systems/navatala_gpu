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

__kernel void navatala_ml_compute_neighbor_counts_f32(__global const float* points, __global const float* epsilon, __global const uint* nPoints, __global const uint* nDims, __global uint* counts) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nPoints[0];
  uint d = nDims[0];
  float eps = epsilon[0];
  float epsSq = (eps * eps);
  bool inBounds = (gid < n);
  if (inBounds) {
    uint iBase = (gid * d);
    uint countAccum = (uint)(0u);
    for (int j = 0; j < (int)(n); ++j) {
      uint jU32 = ((uint)(j));
      uint jBase = (jU32 * d);
      float distSqAccum = as_float(0x00000000u);
      for (int k = 0; k < (int)(d); ++k) {
        uint kU32 = ((uint)(k));
        uint iIdx = (iBase + kU32);
        uint jIdx = (jBase + kU32);
        float iVal = points[iIdx];
        float jVal = points[jIdx];
        float diff = (iVal - jVal);
        float diffSq = (diff * diff);
        float currentDistSq = distSqAccum;
        float newDistSq = (currentDistSq + diffSq);
        distSqAccum = newDistSq;
      }
      float finalDistSq = distSqAccum;
      if (finalDistSq < epsSq) {
        uint currentCount = countAccum;
        uint newCount = (currentCount + (uint)(1u));
        countAccum = newCount;
      }
    }
    uint finalCount = countAccum;
    counts[gid] = finalCount;
  }
}
