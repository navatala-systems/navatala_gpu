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
extern "C" __global__ void navatala_cfd_add_vol_vector(const float* ax, const float* ay, const float* az, const float* bx, const float* by_, const float* bz, const int* params, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0])) {
    return;
  } else {
    outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (ax[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + bx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (ay[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + by_[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (az[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + bz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
  }
}
