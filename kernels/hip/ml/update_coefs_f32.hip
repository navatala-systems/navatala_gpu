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
extern "C" __global__ void navatala_ml_update_coefs_f32(const float* equiangular, const float* gamma, const unsigned int* activeMask, const unsigned int* nCols, float* beta) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int d = nCols[0];
  float gammaVal = gamma[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    unsigned int mask = activeMask[gid];
    bool isActive = (mask == 1u);
    if (isActive) {
      float betaJ = beta[gid];
      float eqJ = equiangular[gid];
      float update = (gammaVal * eqJ);
      float newBeta = (betaJ + update);
      beta[gid] = newBeta;
    }
  }
}
