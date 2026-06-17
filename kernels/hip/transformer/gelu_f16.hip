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
extern "C" __global__ void navatala_transformer_gelu_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    __half x2 = (x * x);
    __half x3 = (x2 * x);
    __half cubeTerm = (__float2half(0.044715f) * x3);
    __half inner = (x + cubeTerm);
    __half tanhArg = (__float2half(0.797885f) * inner);
    __half tanhVal = tanh(tanhArg);
    __half onePlusTanh = (__float2half(1.000000f) + tanhVal);
    __half halfX = (__float2half(0.500000f) * x);
    __half result = (halfX * onePlusTanh);
    _output[gid] = result;
  }
}
