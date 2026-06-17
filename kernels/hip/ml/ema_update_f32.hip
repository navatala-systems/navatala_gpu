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
extern "C" __global__ void navatala_ml_ema_update_f32(const float* x, const float* alpha, float* ema, unsigned int* count) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    float xVal = x[0];
    float a = alpha[0];
    float oldEma = ema[0];
    unsigned int n = count[0];
    float oneMinusA = (__uint_as_float(0x3f800000u) - a);
    float alphaX = (a * xVal);
    float omAEma = (oneMinusA * oldEma);
    float emaUpd = (alphaX + omAEma);
    float newEma = (((n == 0u)) ? (xVal) : (emaUpd));
    ema[0] = newEma;
    unsigned int newCnt = (n + 1u);
    count[0] = newCnt;
  }
}
