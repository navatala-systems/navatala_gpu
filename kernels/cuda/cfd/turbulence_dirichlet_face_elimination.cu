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
extern "C" __global__ void navatala_cfd_turbulence_dirichlet_face_elimination(const int* owner, const int* neighbour, const int* constrainedMask, const float* constrainedValue, const int* nIntFaces, float* upper, float* lower, float* source) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= nIntFaces[0])) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int oC = constrainedMask[o];
    int nC = constrainedMask[n];
    if (((oC | nC) == 0)) {
      return;
    } else {
      float up = upper[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float lo = lower[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if ((oC != 0)) {
        float cv = constrainedValue[o];
        atomicAdd(&source[n], (-(lo * cv)));
      }
      if ((nC != 0)) {
        float cv = constrainedValue[n];
        atomicAdd(&source[o], (-(up * cv)));
      }
      upper[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
      lower[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    }
  }
}
