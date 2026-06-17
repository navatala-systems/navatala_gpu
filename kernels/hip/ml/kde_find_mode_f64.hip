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
extern "C" __global__ void navatala_ml_kde_find_mode_f64(const double* densities, const double* queryPoints, const unsigned int* m, double* modeValue, double* modeLocation) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if ((gid == 0u)) {
    unsigned int numPoints = m[0];
    double maxDensity = densities[0];
    double maxLocation = queryPoints[0];
    for (int i = 0; i < (int)(numPoints); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      double d = densities[iU32];
      double loc = queryPoints[iU32];
      double currMaxDensity = maxDensity;
      double currMaxLocation = maxLocation;
      bool isGreater = (d > currMaxDensity);
      double newMaxDensity = ((isGreater) ? (d) : (currMaxDensity));
      double newMaxLocation = ((isGreater) ? (loc) : (currMaxLocation));
      maxDensity = newMaxDensity;
      maxLocation = newMaxLocation;
    }
    double finalMaxDensity = maxDensity;
    double finalMaxLocation = maxLocation;
    modeValue[0] = finalMaxDensity;
    modeLocation[0] = finalMaxLocation;
  }
}
