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
extern "C" __global__ void navatala_ml_init_embedding_random_f64(const unsigned int* nPoints, const unsigned int* outputDims, const unsigned int* seed, double* Y) {
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
    double maxVal = __longlong_as_double(0x41dfffffffc00000ull);
    double u1 = (((double)(state1)) / maxVal);
    double u2 = (((double)(state2)) / maxVal);
    double negTwoLnU1 = (__longlong_as_double(0xc000000000000000ull) * log(u1));
    double sqrtTerm = sqrt(negTwoLnU1);
    double twoPiU2 = (__longlong_as_double(0x401921fb54411744ull) * u2);
    double z = (sqrtTerm * cos(twoPiU2));
    double yVal = (z * __longlong_as_double(0x3f847ae147ae147bull));
    Y[gid] = yVal;
  }
}
