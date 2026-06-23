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
extern "C" __global__ void navatala_cfd_pressure_sn_grad(const float* phiHbyA, const int* ownerAll, const float* sfX, const float* sfY, const float* sfZ, const float* magSf, const float* ux, const float* uy, const float* uz, const float* uBcX, const float* uBcY, const float* uBcZ, const unsigned int* uBcMask, const float* rAUf, const unsigned int* snGradMask, const unsigned int* params, float* outSnGrad) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[1]))) {
    return;
  } else {
    unsigned int mask = snGradMask[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (((int)(mask)) == 0) {
      outSnGrad[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
    } else {
      int o = ownerAll[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      if (o < 0) {
        outSnGrad[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
      } else {
        if (o >= ((int)(params[0]))) {
          outSnGrad[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
        } else {
          float uxF = ux[o];
          float uyF = uy[o];
          float uzF = uz[o];
          unsigned int bcm = uBcMask[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
          if (((int)(bcm)) == 2) {
            uxF = uBcX[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
            uyF = uBcY[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
            uzF = uBcZ[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
          }
          float phiU = (((sfX[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * uxF) + (sfY[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * uyF)) + (sfZ[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * uzF));
          float rf = rAUf[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
          float ms = magSf[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
          float denom = (rf * ms);
          float ph = phiHbyA[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
          if (abs(denom) < __uint_as_float(0x1e3ce508u)) {
            outSnGrad[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x00000000u);
          } else {
            outSnGrad[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = ((ph - (rf * phiU)) / denom);
          }
        }
      }
    }
  }
}
