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
extern "C" __global__ void navatala_cfd_primitives_phi_from_u(const int* owner, const int* neighbour, const float* weights, const float* sfX, const float* sfY, const float* sfZ, const float* ux, const float* uy, const float* uz, const float* bcX, const float* bcY, const float* bcZ, const unsigned int* bcMask, const int* params, float* outPhi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float uox = ux[o];
    float uoy = uy[o];
    float uoz = uz[o];
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1]) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float iw = (__uint_as_float(0x3f800000u) - w);
      float ufx = ((w * uox) + (iw * ux[n]));
      float ufy = ((w * uoy) + (iw * uy[n]));
      float ufz = ((w * uoz) + (iw * uz[n]));
      outPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufx) + (sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufy)) + (sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufz));
    } else {
      if (bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] == 1u) {
        float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float iw = (__uint_as_float(0x3f800000u) - w);
        float ufx = ((w * uox) + (iw * bcX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
        float ufy = ((w * uoy) + (iw * bcY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
        float ufz = ((w * uoz) + (iw * bcZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
        outPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufx) + (sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufy)) + (sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufz));
      } else {
        if (bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] == 2u) {
          float ufx = bcX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
          float ufy = bcY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
          float ufz = bcZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
          outPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufx) + (sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufy)) + (sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * ufz));
        } else {
          outPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((sfX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * uox) + (sfY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * uoy)) + (sfZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * uoz));
        }
      }
    }
  }
}
