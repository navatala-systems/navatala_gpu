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
extern "C" __global__ void navatala_transformer_dropout_forward_f16(const __half* _input, const unsigned int* count, const float* prob, const unsigned int* seed, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  float p = prob[0u];
  unsigned int s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half xF16 = _input[gid];
    float x = ((float)(xF16));
    unsigned int h0 = (gid ^ s);
    unsigned int h1 = (h0 * 3432918353u);
    unsigned int h2 = (h1 ^ (h1 >> 15u));
    unsigned int h3 = (h2 * 461845907u);
    unsigned int hashVal = (h3 ^ (h3 >> 13u));
    float randF32 = (((float)(hashVal)) / __uint_as_float(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (__uint_as_float(0x3f800000u) - p);
    float scale = (__uint_as_float(0x3f800000u) / oneMinusP);
    float scaled = (x * scale);
    float resultF32 = ((keep) ? (scaled) : (__uint_as_float(0x00000000u)));
    __half result = ((__half)(resultF32));
    _output[gid] = result;
  }
}
