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
extern "C" __global__ void navatala_ml_make_sparse_uncorrelated_f32(const float* randomNormal, const unsigned int* sparseMask, const float* scale, const unsigned int* n, const unsigned int* d, float* X) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0];
  unsigned int numD = d[0];
  unsigned int totalElems = (numN * numD);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int featureIdx = (gid % numD);
    unsigned int maskVal = sparseMask[gid];
    float randVal = randomNormal[gid];
    float scaleVal = scale[featureIdx];
    bool isNonZero = (maskVal == 1u);
    float scaledRand = (randVal * scaleVal);
    float result = ((isNonZero) ? (scaledRand) : (__uint_as_float(0x00000000u)));
    X[gid] = result;
  }
}
