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
extern "C" __global__ void navatala_ml_softmax_bf16(const __nv_bfloat16* _input, const unsigned int* count, __nv_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  unsigned int numIters = ((countVal + 255u) / 256u);
  float gsMax = __uint_as_float(0xff7fc99eu);
  for (int itA = 0; itA < (int)(numIters); ++itA) {
    unsigned int idxA = (lid + (((unsigned int)(itA)) * 256u));
    if (idxA < countVal) {
      float xA = ((float)(_input[idxA]));
      gsMax = (((gsMax > xA)) ? (gsMax) : (xA));
    }
  }
  sdata[lid] = gsMax;
  __syncthreads();
  unsigned int maxStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = maxStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = maxStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    maxStride = nextStride;
    __syncthreads();
  }
  float maxVal = sdata[0];
  __syncthreads();
  float gsSum = __uint_as_float(0x00000000u);
  for (int itB = 0; itB < (int)(numIters); ++itB) {
    unsigned int idxB = (lid + (((unsigned int)(itB)) * 256u));
    if (idxB < countVal) {
      float eB = exp((((float)(_input[idxB])) - maxVal));
      gsSum = (gsSum + eB);
    }
  }
  sdata[lid] = gsSum;
  __syncthreads();
  unsigned int sumStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = sumStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = sumStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    sumStride = nextStride;
    __syncthreads();
  }
  float sumVal = sdata[0];
  for (int itC = 0; itC < (int)(numIters); ++itC) {
    unsigned int idxC = (lid + (((unsigned int)(itC)) * 256u));
    if (idxC < countVal) {
      float eC = exp((((float)(_input[idxC])) - maxVal));
      _output[idxC] = ((__nv_bfloat16)((eC / sumVal)));
    }
  }
}
