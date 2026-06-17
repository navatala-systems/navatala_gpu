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
extern "C" __global__ void navatala_ml_resize_bilinear_f16(const __half* _input, const unsigned int* inSize, const unsigned int* outSize, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int outSizeVal = outSize[0];
  if ((gid < outSizeVal)) {
    unsigned int inSizeVal = inSize[0];
    unsigned int inM1 = (inSizeVal - 1u);
    unsigned int outM1 = (outSizeVal - 1u);
    float pos = ((((float)(gid)) * ((float)(inM1))) / ((float)(outM1)));
    float i0f = floor(pos);
    unsigned int i0 = ((unsigned int)(i0f));
    float frac = (pos - i0f);
    unsigned int i0p1 = (i0 + 1u);
    unsigned int i1 = (((i0p1 < inM1)) ? (i0p1) : (inM1));
    float v0 = ((float)(_input[i0]));
    float v1 = ((float)(_input[i1]));
    float outF = ((v0 * (__uint_as_float(0x3f800000u) - frac)) + (v1 * frac));
    _output[gid] = ((__half)(outF));
  }
}
