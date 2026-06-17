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
extern "C" __global__ void navatala_cfd_primitives_sn_grad_scalar(const float* cellVal, const int* owner, const int* neighbour, const float* delta, const float* faceVal, const unsigned int* faceMask, const int* params, float* outSnGrad) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float aO = cellVal[o];
    if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1])) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float aN = cellVal[n];
      outSnGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (aN - aO));
    } else {
      if ((faceMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0u)) {
        float aN = faceVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        outSnGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (aN - aO));
      } else {
        outSnGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
      }
    }
  }
}
