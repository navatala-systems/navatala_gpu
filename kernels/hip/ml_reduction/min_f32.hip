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
extern "C" __global__ void navatala_ml_reduction_min_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsAcc = __uint_as_float(0x7f7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    unsigned int idx = (lid + (((unsigned int)(it)) * 256u));
    if ((idx < countVal)) {
      float raw = _input[idx];
      float v = raw;
      gsAcc = (((gsAcc < v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  __syncthreads();
  unsigned int redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine < other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = redStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    float outV = finalF;
    result[0] = outV;
  }
}
