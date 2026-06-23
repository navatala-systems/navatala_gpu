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
extern "C" __global__ void navatala_cfd_primitives_phi_from_u_internal(const int* owner, const int* neighbour, const float* weights, const float* sfX, const float* sfY, const float* sfZ, const float* ux, const float* uy, const float* uz, const unsigned int* params, float* outPhi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[1]))) {
    return;
  } else {
    int o = owner[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int n = neighbour[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float w = weights[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    float iw = (__uint_as_float(0x3f800000u) - w);
    float ufx = ((w * ux[o]) + (iw * ux[n]));
    float ufy = ((w * uy[o]) + (iw * uy[n]));
    float ufz = ((w * uz[o]) + (iw * uz[n]));
    outPhi[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (((sfX[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * ufx) + (sfY[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * ufy)) + (sfZ[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * ufz));
  }
}
