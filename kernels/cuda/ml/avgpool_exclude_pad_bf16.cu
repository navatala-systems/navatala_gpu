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
extern "C" __global__ void navatala_ml_avgpool_exclude_pad_bf16(const __nv_bfloat16* _input, const unsigned int* window, const unsigned int* stride, const unsigned int* inCount, const unsigned int* outCount, __nv_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int outCountVal = outCount[0];
  if (gid < outCountVal) {
    unsigned int windowVal = window[0];
    unsigned int strideVal = stride[0];
    unsigned int inCountVal = inCount[0];
    unsigned int base = (gid * strideVal);
    float sum = __uint_as_float(0x00000000u);
    float cnt = __uint_as_float(0x00000000u);
    for (int w = 0; w < (int)(windowVal); ++w) {
      unsigned int idx = (base + ((unsigned int)(w)));
      if (idx < inCountVal) {
        float v = ((float)(_input[idx]));
        sum = (sum + v);
        cnt = (cnt + __uint_as_float(0x3f800000u));
      }
    }
    _output[gid] = ((__nv_bfloat16)((sum / (((cnt > __uint_as_float(0x00000000u))) ? (cnt) : (__uint_as_float(0x3f800000u))))));
  }
}
