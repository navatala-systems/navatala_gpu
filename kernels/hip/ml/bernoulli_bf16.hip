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
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_ml_bernoulli_bf16(const unsigned int* count, const unsigned int* seed, const float* p, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  unsigned int seedVal = seed[0];
  if (gid < countVal) {
    unsigned int uk0 = ((seedVal + (gid * 2654435769u)) + 0u);
    unsigned int uk1 = ((uk0 ^ (uk0 >> 16u)) * 2246822519u);
    unsigned int uk2 = ((uk1 ^ (uk1 >> 13u)) * 3266489917u);
    unsigned int uk3 = (uk2 ^ (uk2 >> 16u));
    float uF = (((float)(uk3)) * __uint_as_float(0x2f800000u));
    float pVal = p[0];
    float bit = (((uF < pVal)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
    __hip_bfloat16 outV = ((__hip_bfloat16)(bit));
    _output[gid] = outV;
  }
}
