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
extern "C" __global__ void navatala_cfd_bc_sn_grad_face_flux(const float* cf, const float* delta, const float* bcSnGrad, const int* bcSnGradMask, const int* counts, float* outFlux) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1])) {
    return;
  } else {
    float d = delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float sg = bcSnGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float flux = __uint_as_float(0x00000000u);
    if ((bcSnGradMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0)) {
      if ((d != __uint_as_float(0x00000000u))) {
        flux = (cf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (sg / d));
      }
    }
    outFlux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = flux;
  }
}
