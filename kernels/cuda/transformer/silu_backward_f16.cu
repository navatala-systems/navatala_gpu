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
#include <cuda_fp16.h>
extern "C" __global__ void navatala_transformer_silu_backward_f16(const __half* _input, const __half* gradOutput, const unsigned int* count, __half* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    __half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float negX = (-xF32);
    float expNegX = exp(negX);
    float denom = (__uint_as_float(0x3f800000u) + expNegX);
    float sigmoid = (__uint_as_float(0x3f800000u) / denom);
    float oneMinusSigmoid = (__uint_as_float(0x3f800000u) - sigmoid);
    float xTimesOneMinusSig = (xF32 * oneMinusSigmoid);
    float bracketTerm = (__uint_as_float(0x3f800000u) + xTimesOneMinusSig);
    float siluPrime = (sigmoid * bracketTerm);
    float dxF32 = (dyF32 * siluPrime);
    __half dx = ((__half)(dxF32));
    gradInput[gid] = dx;
  }
}
