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
extern "C" __global__ void navatala_cfd_vof_alpha_phi_all(const float* alphaPhiInt, const float* phiAll, const float* alphaF, const int* params, float* alphaPhiAllOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0])) {
    return;
  } else {
    if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1])) {
      alphaPhiAllOut[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = alphaPhiInt[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    } else {
      float a = alphaF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if ((a < __uint_as_float(0x00000000u))) {
        a = __uint_as_float(0x00000000u);
      }
      if ((a > __uint_as_float(0x3f800000u))) {
        a = __uint_as_float(0x3f800000u);
      }
      alphaPhiAllOut[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (phiAll[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * a);
    }
  }
}
