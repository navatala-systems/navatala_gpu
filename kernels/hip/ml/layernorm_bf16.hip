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
extern "C" __global__ void navatala_ml_layernorm_bf16(const __hip_bfloat16* x, const __hip_bfloat16* gamma, const __hip_bfloat16* beta, const unsigned int* count, const float* eps, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  float n = ((float)(countVal));
  unsigned int numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gsM = __uint_as_float(0x00000000u);
  for (int itM = 0; itM < (int)(numIters); ++itM) {
    if ((lid + (((unsigned int)(itM)) * 256u)) < countVal) {
      gsM = (gsM + ((float)(x[(lid + (((unsigned int)(itM)) * 256u))])));
    }
  }
  sdata[lid] = gsM;
  __syncthreads();
  unsigned int meanStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = meanStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = meanStr;
    unsigned int nextStride = (strideToHalve >> 1u);
    meanStr = nextStride;
    __syncthreads();
  }
  float mean = (sdata[0] / n);
  __syncthreads();
  float gsV = __uint_as_float(0x00000000u);
  for (int itV = 0; itV < (int)(numIters); ++itV) {
    if ((lid + (((unsigned int)(itV)) * 256u)) < countVal) {
      gsV = (gsV + ((((float)(x[(lid + (((unsigned int)(itV)) * 256u))])) - mean) * (((float)(x[(lid + (((unsigned int)(itV)) * 256u))])) - mean)));
    }
  }
  sdata[lid] = gsV;
  __syncthreads();
  unsigned int varStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = varStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = varStr;
    unsigned int nextStride = (strideToHalve >> 1u);
    varStr = nextStride;
    __syncthreads();
  }
  float var = (sdata[0] / n);
  float denom = sqrt((var + epsVal));
  for (int itW = 0; itW < (int)(numIters); ++itW) {
    unsigned int idxW = (lid + (((unsigned int)(itW)) * 256u));
    if (idxW < countVal) {
      float xnW = ((((float)(x[idxW])) - mean) / denom);
      float outF = ((((float)(gamma[idxW])) * xnW) + ((float)(beta[idxW])));
      _output[idxW] = ((__hip_bfloat16)(outF));
    }
  }
}
