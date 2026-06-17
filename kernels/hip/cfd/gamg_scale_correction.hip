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
extern "C" __global__ void navatala_cfd_gamg_scale_correction(const float* z, const float* r, const float* Az, const float* diagInv, const float* sfPtr, const unsigned int* nRows, float* corr) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(nRows[0])))) {
    return;
  } else {
    float sf = sfPtr[0];
    float zi = z[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float ri = r[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float azi = Az[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float di = diagInv[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float sfz = (sf * zi);
    float sfaz = (sf * azi);
    float residual = (ri - sfaz);
    float corrVal = (sfz + (residual * di));
    corr[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = corrVal;
  }
}
