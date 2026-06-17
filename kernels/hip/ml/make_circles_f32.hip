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
extern "C" __global__ void navatala_ml_make_circles_f32(const unsigned int* labels, const float* randomUniform, const float* randomNormal, const float* noise, const float* factor, const unsigned int* n, float* data) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    float sigma = noise[0];
    float innerFactor = factor[0];
    unsigned int label = labels[gid];
    float u = randomUniform[gid];
    float theta = (__uint_as_float(0x40c90fdbu) * u);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    unsigned int outIdxX = (gid * 2u);
    unsigned int outIdxY = (outIdxX + 1u);
    float noiseX = randomNormal[outIdxX];
    float noiseY = randomNormal[outIdxY];
    bool isOuterCircle = (label == 1u);
    float radius = ((isOuterCircle) ? (__uint_as_float(0x3f800000u)) : (innerFactor));
    float baseX = (radius * cosTheta);
    float baseY = (radius * sinTheta);
    float resultX = (baseX + (sigma * noiseX));
    float resultY = (baseY + (sigma * noiseY));
    data[outIdxX] = resultX;
    data[outIdxY] = resultY;
  }
}
