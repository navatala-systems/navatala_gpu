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
extern "C" __global__ void navatala_ml_resize_nearest_bf16(const __hip_bfloat16* _input, const unsigned int* inSize, const unsigned int* outSize, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int outSizeVal = outSize[0];
  if (gid < outSizeVal) {
    unsigned int inSizeVal = inSize[0];
    unsigned int idx = ((gid * inSizeVal) / outSizeVal);
    __hip_bfloat16 v = _input[idx];
    _output[gid] = v;
  }
}
