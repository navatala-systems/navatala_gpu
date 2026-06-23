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
extern "C" __global__ void navatala_dataframe_p2_update_f32(const float* x, float* q0, float* q1, float* q2, float* q3, float* q4, unsigned int* pos0, unsigned int* pos1, unsigned int* pos2, unsigned int* pos3, unsigned int* pos4, const float* targetP, unsigned int* count) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    float xVal = x[0];
    unsigned int n = count[0];
    float pVal = targetP[0];
    float h0 = q0[0];
    float h1 = q1[0];
    float h2 = q2[0];
    float h3 = q3[0];
    float h4 = q4[0];
    float newH0 = (((n == 0u)) ? (xVal) : ((((xVal < h0)) ? (xVal) : (h0))));
    float finalH0 = (((n >= 5u)) ? ((((xVal < h0)) ? (xVal) : (h0))) : (newH0));
    float finalH4 = (((n >= 5u)) ? ((((xVal > h4)) ? (xVal) : (h4))) : ((((n == 4u)) ? (xVal) : (h4))));
    float alpha = __uint_as_float(0x3c23d70au);
    float oneMinusAlpha = (__uint_as_float(0x3f800000u) - alpha);
    float finalH2 = (((n >= 5u)) ? (((oneMinusAlpha * h2) + (alpha * xVal))) : ((((n == 2u)) ? (xVal) : (h2))));
    q0[0] = finalH0;
    q2[0] = finalH2;
    q4[0] = finalH4;
    unsigned int newCount = (n + 1u);
    count[0] = newCount;
  }
}
