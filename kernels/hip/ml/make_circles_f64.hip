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
extern "C" __global__ void navatala_ml_make_circles_f64(const unsigned int* labels, const double* randomUniform, const double* randomNormal, const double* noise, const double* factor, const unsigned int* n, double* data) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    double sigma = noise[0];
    double innerFactor = factor[0];
    unsigned int label = labels[gid];
    double u = randomUniform[gid];
    double theta = (__longlong_as_double(0x401921fb54442d18ull) * u);
    double cosTheta = cos(theta);
    double sinTheta = sin(theta);
    unsigned int outIdxX = (gid * 2u);
    unsigned int outIdxY = (outIdxX + 1u);
    double noiseX = randomNormal[outIdxX];
    double noiseY = randomNormal[outIdxY];
    bool isOuterCircle = (label == 1u);
    double radius = ((isOuterCircle) ? (__longlong_as_double(0x3ff0000000000000ull)) : (innerFactor));
    double baseX = (radius * cosTheta);
    double baseY = (radius * sinTheta);
    double resultX = (baseX + (sigma * noiseX));
    double resultY = (baseY + (sigma * noiseY));
    data[outIdxX] = resultX;
    data[outIdxY] = resultY;
  }
}
