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
extern "C" __global__ void navatala_transformer_bias_gelu_f32(const float* _input, const float* bias, const unsigned int* count, const unsigned int* biasSize, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int bSize = biasSize[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    unsigned int biasIdx = (gid % bSize);
    float b = bias[biasIdx];
    float xBias = (x + b);
    float x2 = (xBias * xBias);
    float x3 = (x2 * xBias);
    float cubeTerm = (__uint_as_float(0x3d372713u) * x3);
    float inner = (xBias + cubeTerm);
    float tanhArg = (__uint_as_float(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (__uint_as_float(0x3f800000u) + tanhVal);
    float halfXBias = (__uint_as_float(0x3f000000u) * xBias);
    float result = (halfXBias * onePlusTanh);
    _output[gid] = result;
  }
}
