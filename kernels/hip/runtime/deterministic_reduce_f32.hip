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
extern "C" __global__ void navatala_runtime_deterministic_reduce_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float val = _input[gid];
    sdata[lid] = val;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int f32RedStride = 128u;
  for (int f32RedStep = 0; f32RedStep < (int)(8); ++f32RedStep) {
    unsigned int f32Stride = f32RedStride;
    if (lid < f32Stride) {
      float other = sdata[(lid + f32Stride)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int f32StrideToHalve = f32RedStride;
    unsigned int f32NextStride = (f32StrideToHalve >> 1u);
    f32RedStride = f32NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float totalSum = sdata[0];
    result[0] = totalSum;
  }
}
