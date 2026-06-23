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
extern "C" __global__ void navatala_ml_normal_f32(const unsigned int* count, const unsigned int* seed, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  unsigned int seedVal = seed[0];
  if (gid < countVal) {
    unsigned int ak0 = ((seedVal + (gid * 2654435769u)) + 0u);
    unsigned int ak1 = ((ak0 ^ (ak0 >> 16u)) * 2246822519u);
    unsigned int ak2 = ((ak1 ^ (ak1 >> 13u)) * 3266489917u);
    unsigned int ak3 = (ak2 ^ (ak2 >> 16u));
    float u1 = (((float)(ak3)) * __uint_as_float(0x2f800000u));
    unsigned int bk0 = ((seedVal + (gid * 2654435769u)) + 2246822519u);
    unsigned int bk1 = ((bk0 ^ (bk0 >> 16u)) * 2246822519u);
    unsigned int bk2 = ((bk1 ^ (bk1 >> 13u)) * 3266489917u);
    unsigned int bk3 = (bk2 ^ (bk2 >> 16u));
    float u2 = (((float)(bk3)) * __uint_as_float(0x2f800000u));
    float u1e = (u1 + __uint_as_float(0x33d6bf95u));
    float r = sqrt((__uint_as_float(0xc0000000u) * log(u1e)));
    float z = (r * cos((__uint_as_float(0x40c90fdbu) * u2)));
    float outV = z;
    _output[gid] = outV;
  }
}
