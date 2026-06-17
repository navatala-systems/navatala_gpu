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
extern "C" __global__ void navatala_dataframe_normal_box_muller_f32(unsigned long long* state, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0u];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    unsigned long long s = state[gid];
    unsigned long long aU64 = ((unsigned long long)(1103515245u));
    unsigned long long cU64 = ((unsigned long long)(12345u));
    unsigned long long s1 = (((aU64 * s) + cU64) & ((unsigned long long)(2147483647u)));
    float u1 = (((float)(s1)) / __uint_as_float(0x4f000000u));
    unsigned long long s2 = (((aU64 * s1) + cU64) & ((unsigned long long)(2147483647u)));
    float u2 = (((float)(s2)) / __uint_as_float(0x4f000000u));
    state[gid] = s2;
    float u1Clamped = (((u1 < __uint_as_float(0x33d6bf95u))) ? (__uint_as_float(0x33d6bf95u)) : (u1));
    float logU1 = log(u1Clamped);
    float negTwoLogU1 = (__uint_as_float(0xc0000000u) * logU1);
    float sqrtTerm = sqrt(negTwoLogU1);
    float angle = (__uint_as_float(0x40c90fdbu) * u2);
    float cosTerm = cos(angle);
    float z = (sqrtTerm * cosTerm);
    _output[gid] = z;
  }
}
