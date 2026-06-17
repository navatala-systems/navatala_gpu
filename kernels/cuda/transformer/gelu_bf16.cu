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
#include <cuda_bf16.h>
extern "C" __global__ void navatala_transformer_gelu_bf16(const __nv_bfloat16* _input, const unsigned int* count, __nv_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __nv_bfloat16 xBF16 = _input[gid];
    float x = ((float)(xBF16));
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (__uint_as_float(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float tanhArg = (__uint_as_float(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (__uint_as_float(0x3f800000u) + tanhVal);
    float halfX = (__uint_as_float(0x3f000000u) * x);
    float resultF32 = (halfX * onePlusTanh);
    __nv_bfloat16 result = ((__nv_bfloat16)(resultF32));
    _output[gid] = result;
  }
}
