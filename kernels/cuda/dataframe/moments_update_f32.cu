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
extern "C" __global__ void navatala_dataframe_moments_update_f32(const float* x, unsigned int* n, float* M1, float* M2, float* M3, float* M4) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    unsigned int nOld = n[0];
    float m1Old = M1[0];
    float m2Old = M2[0];
    float m3Old = M3[0];
    float m4Old = M4[0];
    float xVal = x[0];
    unsigned int nNew = (nOld + 1u);
    float nNewF = ((float)(nNew));
    float nOldF = ((float)(nOld));
    float delta = (xVal - m1Old);
    float delta_n = (delta / nNewF);
    float delta_n2 = (delta_n * delta_n);
    float term1 = ((delta * delta_n) * nOldF);
    float m1New = (m1Old + delta_n);
    float nNew2 = (nNewF * nNewF);
    float threeNNew = (__uint_as_float(0x40400000u) * nNewF);
    float nNew2MinusThreeN = (nNew2 - threeNNew);
    float m4Coeff = (nNew2MinusThreeN + __uint_as_float(0x40400000u));
    float m4Term1 = ((term1 * delta_n2) * m4Coeff);
    float m4Term2 = ((__uint_as_float(0x40c00000u) * delta_n2) * m2Old);
    float m4Term3 = ((__uint_as_float(0x40800000u) * delta_n) * m3Old);
    float m4New = (((m4Old + m4Term1) + m4Term2) - m4Term3);
    float nNewMinus2 = (nNewF - __uint_as_float(0x40000000u));
    float m3Term1 = ((term1 * delta_n) * nNewMinus2);
    float m3Term2 = ((__uint_as_float(0x40400000u) * delta_n) * m2Old);
    float m3New = ((m3Old + m3Term1) - m3Term2);
    float m2New = (m2Old + term1);
    n[0] = nNew;
    M1[0] = m1New;
    M2[0] = m2New;
    M3[0] = m3New;
    M4[0] = m4New;
  }
}
