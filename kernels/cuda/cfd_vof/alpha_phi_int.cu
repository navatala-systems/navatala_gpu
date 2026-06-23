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
extern "C" __global__ void navatala_cfd_vof_alpha_phi_int(const float* alpha, const float* phi, const int* owner, const int* nei, const float* phir, const float* alphaF, const int* faceParams, float* outAlphaPhi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= faceParams[0]) {
    return;
  } else {
    float ph = phi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float a = __uint_as_float(0x00000000u);
    if (ph >= __uint_as_float(0x00000000u)) {
      a = alpha[o];
    } else {
      a = alpha[n];
    }
    if (a < __uint_as_float(0x00000000u)) {
      a = __uint_as_float(0x00000000u);
    }
    if (a > __uint_as_float(0x3f800000u)) {
      a = __uint_as_float(0x3f800000u);
    }
    float ac = alphaF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (ac < __uint_as_float(0x00000000u)) {
      ac = __uint_as_float(0x00000000u);
    }
    if (ac > __uint_as_float(0x3f800000u)) {
      ac = __uint_as_float(0x3f800000u);
    }
    float oneMinus = (__uint_as_float(0x3f800000u) - ac);
    float comp = (ac * oneMinus);
    outAlphaPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((ph * a) + (phir[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * comp));
  }
}
