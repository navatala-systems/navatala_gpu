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
extern "C" __global__ void navatala_transformer_layer_norm_multi_pass_f16(const __half* _input, const __half* gamma, const __half* beta, const float* epsilon, const unsigned int* batchSize, const unsigned int* hiddenSize, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  float eps = epsilon[0u];
  __shared__ float sumBuf[256];
  __shared__ float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  unsigned int baseIdx = (batchIdx * hs);
  float partialSum = __uint_as_float(0x00000000u);
  float partialSumSq = __uint_as_float(0x00000000u);
  unsigned int iterIdx = lid;
  unsigned int workgroupSize = 256u;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    unsigned int globalIdx = (baseIdx + iterIdx);
    __half xF16 = ((batchValid) ? (_input[globalIdx]) : (__float2half(0.000000f)));
    float xVal = ((float)(xF16));
    partialSum = (partialSum + xVal);
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  sumSqBuf[lid] = partialSumSq;
  __syncthreads();
  bool shouldReduce_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sumBuf_128) {
    unsigned int neighborIdx_sumBuf_128 = (lid + 128u);
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sumBuf_64) {
    unsigned int neighborIdx_sumBuf_64 = (lid + 64u);
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sumBuf_32) {
    unsigned int neighborIdx_sumBuf_32 = (lid + 32u);
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sumBuf_16) {
    unsigned int neighborIdx_sumBuf_16 = (lid + 16u);
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sumBuf_8) {
    unsigned int neighborIdx_sumBuf_8 = (lid + 8u);
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sumBuf_4) {
    unsigned int neighborIdx_sumBuf_4 = (lid + 4u);
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sumBuf_2) {
    unsigned int neighborIdx_sumBuf_2 = (lid + 2u);
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sumBuf_1) {
    unsigned int neighborIdx_sumBuf_1 = (lid + 1u);
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  __syncthreads();
  float totalSum = sumBuf[0u];
  __syncthreads();
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    unsigned int neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    unsigned int neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    unsigned int neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    unsigned int neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    unsigned int neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    unsigned int neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    unsigned int neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    unsigned int neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  __syncthreads();
  float totalSumSq = sumSqBuf[0u];
  float hsF32 = ((float)(hs));
  float mean = (totalSum / hsF32);
  float meanSq = (mean * mean);
  float avgSumSq = (totalSumSq / hsF32);
  float variance = (avgSumSq - meanSq);
  float varEps = (variance + eps);
  float std = sqrt(varEps);
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    if (batchValid) {
      unsigned int globalIdx2 = (baseIdx + iterIdx);
      __half xF16_2 = _input[globalIdx2];
      float xVal2 = ((float)(xF16_2));
      __half gF16 = gamma[iterIdx];
      __half bF16 = beta[iterIdx];
      float g = ((float)(gF16));
      float b = ((float)(bF16));
      float xNorm = ((xVal2 - mean) / std);
      float scaled = (g * xNorm);
      float resultF32 = (scaled + b);
      __half result = ((__half)(resultF32));
      _output[globalIdx2] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}
