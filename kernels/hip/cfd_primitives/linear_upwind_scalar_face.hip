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
extern "C" __global__ void navatala_cfd_primitives_linear_upwind_scalar_face(const float* flux, const float* xCell, const float* gradX, const float* gradY, const float* gradZ, const float* limiter, const int* owner, const int* nei, const float* cx, const float* cy, const float* cz, const float* cfx, const float* cfy, const float* cfz, const float* bcValue, const unsigned int* bcMask, const unsigned int* params, float* outFace) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[0])))) {
    return;
  } else {
    float phif = flux[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int own = owner[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    if (((int)(blockIdx.x * blockDim.x + threadIdx.x) < ((int)(params[1])))) {
      if ((phif >= __uint_as_float(0x00000000u))) {
        float dx = (cfx[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - cx[own]);
        float dy = (cfy[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - cy[own]);
        float dz = (cfz[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - cz[own]);
        float corr = (((dx * gradX[own]) + (dy * gradY[own])) + (dz * gradZ[own]));
        outFace[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (xCell[own] + (limiter[own] * corr));
      } else {
        int neiCell = nei[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
        float dx = (cfx[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - cx[neiCell]);
        float dy = (cfy[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - cy[neiCell]);
        float dz = (cfz[(int)(blockIdx.x * blockDim.x + threadIdx.x)] - cz[neiCell]);
        float corr = (((dx * gradX[neiCell]) + (dy * gradY[neiCell])) + (dz * gradZ[neiCell]));
        outFace[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (xCell[neiCell] + (limiter[neiCell] * corr));
      }
    } else {
      if ((phif >= __uint_as_float(0x00000000u))) {
        outFace[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = xCell[own];
      } else {
        unsigned int mask = bcMask[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
        if ((((int)(mask)) == 0)) {
          outFace[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = xCell[own];
        } else {
          outFace[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = bcValue[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
        }
      }
    }
  }
}
