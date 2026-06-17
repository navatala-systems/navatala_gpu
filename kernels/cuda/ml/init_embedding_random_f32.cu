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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_ml_init_embedding_random_f32(const unsigned int* nPoints, const unsigned int* outputDims, const unsigned int* seed, float* Y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nPoints[0];
  unsigned int d = outputDims[0];
  unsigned int s = seed[0];
  unsigned int totalElems = (n * d);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int state = (s + gid);
    unsigned int lcgA = 48271u;
    unsigned int state1 = (lcgA * state);
    unsigned int state2 = (lcgA * state1);
    float maxVal = __uint_as_float(0x4f000000u);
    float u1 = (((float)(state1)) / maxVal);
    float u2 = (((float)(state2)) / maxVal);
    float negTwoLnU1 = (__uint_as_float(0xc0000000u) * log(u1));
    float sqrtTerm = sqrt(negTwoLnU1);
    float twoPiU2 = (__uint_as_float(0x40c90fdbu) * u2);
    float z = (sqrtTerm * cos(twoPiU2));
    float yVal = (z * __uint_as_float(0x3c23d70au));
    Y[gid] = yVal;
  }
}
