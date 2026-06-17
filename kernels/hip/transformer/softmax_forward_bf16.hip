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
extern "C" __global__ void navatala_transformer_softmax_forward_bf16(const __hip_bfloat16* _input, const unsigned int* batchSize, const unsigned int* seqLen, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  __shared__ float maxBuf[256];
  __shared__ float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  unsigned int globalIdx = ((batchIdx * sl) + lid);
  __hip_bfloat16 xBF16 = (((batchValid && seqValid)) ? (_input[globalIdx]) : (__float2bfloat16(0.000000f)));
  float x = ((seqValid) ? (((float)(xBF16))) : (__uint_as_float(0xf149f2cau)));
  maxBuf[lid] = x;
  __syncthreads();
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    unsigned int neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    unsigned int neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    unsigned int neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    unsigned int neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    unsigned int neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    unsigned int neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    unsigned int neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    unsigned int neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  __syncthreads();
  float maxVal = maxBuf[0u];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = ((seqValid) ? (expVal) : (__uint_as_float(0x00000000u)));
  sumBuf[lid] = expValMasked;
  __syncthreads();
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    unsigned int neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    unsigned int neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    unsigned int neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    unsigned int neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    unsigned int neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    unsigned int neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    unsigned int neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    unsigned int neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if ((batchValid && seqValid)) {
    float resultF32 = (expVal / sumExp);
    __hip_bfloat16 result = ((__hip_bfloat16)(resultF32));
    _output[globalIdx] = result;
  }
}
