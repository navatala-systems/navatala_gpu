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
extern "C" __global__ void navatala_ml_kde_evaluate_epanechnikov_f32(const float* data, const float* queryPoints, const float* bandwidth, const unsigned int* n, const unsigned int* m, float* densities) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numQueries = m[0];
  bool inBounds = (gid < numQueries);
  if (inBounds) {
    unsigned int numSamples = n[0];
    float h = bandwidth[0];
    float x = queryPoints[gid];
    float sum = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)(numSamples); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      float xi = data[iU32];
      float diff = (x - xi);
      float u = (diff / h);
      float absU = abs(u);
      float uSq = (u * u);
      float oneMinusUSq = (__uint_as_float(0x3f800000u) - uSq);
      float kernelRaw = (__uint_as_float(0x3f400000u) * oneMinusUSq);
      bool inSupport = (absU <= __uint_as_float(0x3f800000u));
      float kernelVal = ((inSupport) ? (kernelRaw) : (__uint_as_float(0x00000000u)));
      float currSum = sum;
      sum = (currSum + kernelVal);
    }
    float nFloat = ((float)(numSamples));
    float nh = (nFloat * h);
    float finalSum = sum;
    float density = (finalSum / nh);
    densities[gid] = density;
  }
}
