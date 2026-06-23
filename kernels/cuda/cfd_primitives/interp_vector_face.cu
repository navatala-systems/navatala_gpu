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
extern "C" __global__ void navatala_cfd_primitives_interp_vector_face(const int* owner, const int* neighbour, const float* weights, const float* cellX, const float* cellY, const float* cellZ, const float* bcX, const float* bcY, const float* bcZ, const unsigned int* bcMask, const int* params, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float uxo = cellX[o];
    float uyo = cellY[o];
    float uzo = cellZ[o];
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1]) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float uxn = cellX[n];
      float uyn = cellY[n];
      float uzn = cellZ[n];
      float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float iw = (__uint_as_float(0x3f800000u) - w);
      outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * uxo) + (iw * uxn));
      outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * uyo) + (iw * uyn));
      outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * uzo) + (iw * uzn));
    } else {
      unsigned int m = bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if (m == 1u) {
        float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float iw = (__uint_as_float(0x3f800000u) - w);
        outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * uxo) + (iw * bcX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
        outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * uyo) + (iw * bcY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
        outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * uzo) + (iw * bcZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
      } else {
        if (m == 2u) {
          outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = bcX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
          outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = bcY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
          outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = bcZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        } else {
          outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = uxo;
          outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = uyo;
          outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = uzo;
        }
      }
    }
  }
}
