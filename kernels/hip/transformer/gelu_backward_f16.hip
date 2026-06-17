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
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_gelu_backward_f16(const __half* _input, const __half* gradOutput, const unsigned int* count, __half* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    __half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float x2 = (xF32 * xF32);
    float x3 = (x2 * xF32);
    float cubeTerm = (__uint_as_float(0x3d372713u) * x3);
    float inner = (xF32 + cubeTerm);
    float u = (__uint_as_float(0x3f4c422au) * inner);
    float tanhU = tanh(u);
    float tanh2 = (tanhU * tanhU);
    float sech2 = (__uint_as_float(0x3f800000u) - tanh2);
    float threeCoef = __uint_as_float(0x3e095d4fu);
    float derivInner = (__uint_as_float(0x3f800000u) + (threeCoef * x2));
    float term2a = (__uint_as_float(0x3f000000u) * xF32);
    float term2b = (term2a * sech2);
    float term2c = (term2b * __uint_as_float(0x3f4c422au));
    float term2 = (term2c * derivInner);
    float onePlusTanh = (__uint_as_float(0x3f800000u) + tanhU);
    float term1 = (__uint_as_float(0x3f000000u) * onePlusTanh);
    float geluPrime = (term1 + term2);
    float dxF32 = (dyF32 * geluPrime);
    __half dx = ((__half)(dxF32));
    gradInput[gid] = dx;
  }
}
