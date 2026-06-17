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

__kernel void navatala_ml_nearest_neighbor_f32(__global const float* D, __global const uint* n, __global const uint* m, __global uint* indices, __global float* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint mVal = m[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nVal);
  if (inBounds) {
    uint rowBase = (rowIdx * mVal);
    float initDist = D[rowBase];
    float minDist = initDist;
    uint minIdx = (uint)(0u);
    for (int j = 0; j < (int)(mVal); ++j) {
      uint jU32 = ((uint)(j));
      uint colIdx = (rowBase + jU32);
      float dist = D[colIdx];
      float currentMinDist = minDist;
      uint currentMinIdx = minIdx;
      bool isSmaller = (dist < currentMinDist);
      float newMinDist = ((isSmaller) ? (dist) : (currentMinDist));
      uint newMinIdx = ((isSmaller) ? (jU32) : (currentMinIdx));
      minDist = newMinDist;
      minIdx = newMinIdx;
    }
    uint finalMinIdx = minIdx;
    float finalMinDist = minDist;
    indices[rowIdx] = finalMinIdx;
    distances[rowIdx] = finalMinDist;
  }
}
