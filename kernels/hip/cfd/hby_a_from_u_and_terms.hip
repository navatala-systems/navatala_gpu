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
extern "C" __global__ void navatala_cfd_hby_a_from_u_and_terms(const float* ux, const float* uy, const float* uz, const float* lapx, const float* lapy, const float* lapz, const float* divx, const float* divy, const float* divz, const float* extrax, const float* extray, const float* extraz, const float* rau, const int* counts, const int* modeBuf, float* outx, float* outy, float* outz) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0])) {
    return;
  } else {
    float tx = lapx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float ty = lapy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float tz = lapz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (((modeBuf[0] == 1) || (modeBuf[0] == 3))) {
      tx = (tx - divx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      ty = (ty - divy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      tz = (tz - divz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    }
    if (((modeBuf[0] == 2) || (modeBuf[0] == 3))) {
      tx = (tx + extrax[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      ty = (ty + extray[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
      tz = (tz + extraz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    }
    float a = rau[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (ux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (a * tx));
    outy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (uy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (a * ty));
    outz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (uz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (a * tz));
  }
}
