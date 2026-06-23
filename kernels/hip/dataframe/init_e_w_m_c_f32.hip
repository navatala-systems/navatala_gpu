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
extern "C" __global__ void navatala_dataframe_init_e_w_m_c_f32(const float* alphaIn, float* alpha, float* nEff, float* meanX, float* meanY, float* Cxy, float* varX, float* varY) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    float a = alphaIn[0];
    alpha[0] = a;
    nEff[0] = __uint_as_float(0x00000000u);
    meanX[0] = __uint_as_float(0x00000000u);
    meanY[0] = __uint_as_float(0x00000000u);
    Cxy[0] = __uint_as_float(0x00000000u);
    varX[0] = __uint_as_float(0x00000000u);
    varY[0] = __uint_as_float(0x00000000u);
  }
}
