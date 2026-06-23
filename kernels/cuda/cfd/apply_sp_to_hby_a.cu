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
extern "C" __global__ void navatala_cfd_apply_sp_to_hby_a(float* hx, float* hy, float* hz, const float* ux, const float* uy, const float* uz, const float* rAU, const float* sp, const int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float fac = (sp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * rAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    hx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (hx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - (fac * ux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    hy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (hy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - (fac * uy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    hz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (hz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - (fac * uz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
  }
}
