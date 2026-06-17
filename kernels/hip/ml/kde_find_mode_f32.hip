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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_kde_find_mode_f32(const float* densities, const float* queryPoints, const unsigned int* m, float* modeValue, float* modeLocation) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if ((gid == 0u)) {
    unsigned int numPoints = m[0];
    float maxDensity = densities[0];
    float maxLocation = queryPoints[0];
    for (int i = 0; i < (int)(numPoints); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
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
