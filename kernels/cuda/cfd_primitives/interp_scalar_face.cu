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
extern "C" __global__ void navatala_cfd_primitives_interp_scalar_face(const int* owner, const int* neighbour, const float* weights, const float* cellPhi, const float* bcValue, const unsigned int* bcMask, const int* params, float* outPhiF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float po = cellPhi[o];
    if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1]) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float pn = cellPhi[n];
      float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float iw = (__uint_as_float(0x3f800000u) - w);
      outPhiF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * po) + (iw * pn));
    } else {
      if (bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] == 1u) {
        float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float iw = (__uint_as_float(0x3f800000u) - w);
        outPhiF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((w * po) + (iw * bcValue[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
      } else {
        if (bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] == 2u) {
          outPhiF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = bcValue[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        } else {
          outPhiF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = po;
        }
      }
    }
  }
}
