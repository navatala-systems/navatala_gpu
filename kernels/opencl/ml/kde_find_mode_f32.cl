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

__kernel void navatala_ml_kde_find_mode_f32(__global const float* densities, __global const float* queryPoints, __global const uint* m, __global float* modeValue, __global float* modeLocation) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if (gid == (uint)(0u)) {
    uint numPoints = m[0];
    float maxDensity = densities[0];
    float maxLocation = queryPoints[0];
    for (int i = 0; i < (int)(numPoints); ++i) {
      uint iU32 = ((uint)(i));
      float d = densities[iU32];
      float loc = queryPoints[iU32];
      float currMaxDensity = maxDensity;
      float currMaxLocation = maxLocation;
      bool isGreater = (d > currMaxDensity);
      float newMaxDensity = ((isGreater) ? (d) : (currMaxDensity));
      float newMaxLocation = ((isGreater) ? (loc) : (currMaxLocation));
      maxDensity = newMaxDensity;
      maxLocation = newMaxLocation;
    }
    float finalMaxDensity = maxDensity;
    float finalMaxLocation = maxLocation;
    modeValue[0] = finalMaxDensity;
    modeLocation[0] = finalMaxLocation;
  }
}
