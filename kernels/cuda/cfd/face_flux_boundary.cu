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
extern "C" __global__ void navatala_cfd_face_flux_boundary(const float* x, const int* owner, const float* cf, const float* bcVal, const unsigned int* bcMask, const unsigned int* params, float* faceFlux) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[0]))) {
    return;
  } else {
    int gf = (((int)(params[1])) + (int)(blockIdx.x * blockDim.x + threadIdx.x));
    int o = owner[gf];
    float xo = x[o];
    unsigned int m = bcMask[gf];
    int mInt = ((int)(m));
    if (mInt == 4) {
      faceFlux[gf] = bcVal[gf];
    } else {
      float other = xo;
      if (mInt == 1) {
        other = bcVal[gf];
      } else {
        if (mInt == 3) {
          other = bcVal[gf];
        } else {
          if (mInt == 2) {
            other = __uint_as_float(0x00000000u);
          }
        }
      }
      faceFlux[gf] = (cf[gf] * (other - xo));
    }
  }
}
