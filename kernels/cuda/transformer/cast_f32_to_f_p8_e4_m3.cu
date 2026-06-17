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
extern "C" __global__ void navatala_transformer_cast_f32_to_f_p8_e4_m3(const float* _input, const float* scale, const unsigned int* count, unsigned char* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float s = scale[0u];
    float scaled = (x * s);
    float minVal = __uint_as_float(0xc3700000u);
    float maxVal = __uint_as_float(0x43700000u);
    float clamped = ((((((scaled < maxVal)) ? (scaled) : (maxVal)) > minVal)) ? ((((scaled < maxVal)) ? (scaled) : (maxVal))) : (minVal));
    float rounded = round(clamped);
    unsigned char fp8Val = ((unsigned char)(rounded));
    _output[gid] = fp8Val;
  }
}
