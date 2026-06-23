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
extern "C" __global__ void navatala_cfd_primitives_phi_from_u_boundary(const int* owner, const float* weights, const float* sfX, const float* sfY, const float* sfZ, const float* ux, const float* uy, const float* uz, const float* bcx, const float* bcy, const float* bcz, const unsigned int* bcmask, const unsigned int* params, float* outPhi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int f = (((int)(params[1])) + (int)(blockIdx.x * blockDim.x + threadIdx.x));
  if (f >= ((int)(params[0]))) {
    return;
  } else {
    int o = owner[f];
    float uox = ux[o];
    float uoy = uy[o];
    float uoz = uz[o];
    float ufx = uox;
    float ufy = uoy;
    float ufz = uoz;
    unsigned int m = bcmask[f];
    int mInt = ((int)(m));
    if (mInt == 1) {
      float w = weights[f];
      float iw = (__uint_as_float(0x3f800000u) - w);
      ufx = ((w * uox) + (iw * bcx[f]));
      ufy = ((w * uoy) + (iw * bcy[f]));
      ufz = ((w * uoz) + (iw * bcz[f]));
    } else {
      if (mInt == 2) {
        ufx = bcx[f];
        ufy = bcy[f];
        ufz = bcz[f];
      }
    }
    outPhi[f] = (((sfX[f] * ufx) + (sfY[f] * ufy)) + (sfZ[f] * ufz));
  }
}
