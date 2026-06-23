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
extern "C" __global__ void navatala_cfd_vof_interp_scalar_face_all(const float* alpha, const int* owner, const int* nei, const float* w, const float* bcAlphaF, const unsigned int* params, float* alphaF) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(params[0]))) {
    return;
  } else {
    if ((int)(blockIdx.x * blockDim.x + threadIdx.x) < ((int)(params[1]))) {
      int o = owner[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      int n = nei[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      float wf = w[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      float wc = (wf * ((float)((wf > __uint_as_float(0x00000000u)))));
      float a = ((wc * alpha[o]) + ((__uint_as_float(0x3f800000u) - wc) * alpha[n]));
      float ac = (a * ((float)((a > __uint_as_float(0x00000000u)))));
      if (ac > __uint_as_float(0x3f800000u)) {
        alphaF[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x3f800000u);
      } else {
        alphaF[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = ac;
      }
    } else {
      float a = bcAlphaF[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
      float ac = (a * ((float)((a > __uint_as_float(0x00000000u)))));
      if (ac > __uint_as_float(0x3f800000u)) {
        alphaF[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __uint_as_float(0x3f800000u);
      } else {
        alphaF[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = ac;
      }
    }
  }
}
