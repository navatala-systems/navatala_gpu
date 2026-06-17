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
extern "C" __global__ void navatala_cfd_predictor_hby_a_from_source(const float* rAU, const float* sourceX, const float* sourceY, const float* sourceZ, const int* counts1, float* outHbyAX, float* outHbyAY, float* outHbyAZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts1[0])) {
    return;
  } else {
    float r = rAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outHbyAX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (r * sourceX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    outHbyAY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (r * sourceY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    outHbyAZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (r * sourceZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
  }
}
