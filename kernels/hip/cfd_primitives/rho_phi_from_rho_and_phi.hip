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
extern "C" __global__ void navatala_cfd_primitives_rho_phi_from_rho_and_phi(const float* phiAll, const float* rhoCell, const int* owner, const int* neighbour, const float* weights, const float* rhoBcVal, const unsigned int* rhoBcMask, const int* params, float* outRhoPhi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float rhoF = rhoCell[o];
    if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < params[1])) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float w0 = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      float iw = (__uint_as_float(0x3f800000u) - w0);
      rhoF = ((w0 * rhoCell[o]) + (iw * rhoCell[n]));
    } else {
      unsigned int m = rhoBcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      if ((m == 1u)) {
        float w0 = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
        float iw = (__uint_as_float(0x3f800000u) - w0);
        rhoF = ((w0 * rhoCell[o]) + (iw * rhoBcVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
      }
      if ((m == 2u)) {
        rhoF = rhoBcVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      }
    }
    float flux = phiAll[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outRhoPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (rhoF * flux);
  }
}
