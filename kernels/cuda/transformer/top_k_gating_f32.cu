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
extern "C" __global__ void navatala_transformer_top_k_gating_f32(const float* routerLogits, const unsigned int* numTokens, const unsigned int* numExperts, const unsigned int* topK, int* expertIndices, float* expertWeights) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tokenIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nt = numTokens[0u];
  unsigned int ne = numExperts[0u];
  unsigned int k = topK[0u];
  __shared__ float logitsBuf[256];
  __shared__ float expBuf[256];
  __shared__ int indexBuf[256];
  bool tokenValid = (tokenIdx < nt);
  bool expertValid = (lid < ne);
  bool valid = (tokenValid && expertValid);
  unsigned int globalIdx = ((tokenIdx * ne) + lid);
  float logit = ((valid) ? (routerLogits[globalIdx]) : (__uint_as_float(0xf149f2cau)));
  logitsBuf[lid] = logit;
  int lidI32 = ((int)(lid));
  indexBuf[lid] = lidI32;
  __syncthreads();
  bool shouldReduce_max_logitsBuf_128 = (lid < 128u);
  if (shouldReduce_max_logitsBuf_128) {
    unsigned int neighborIdx_max_logitsBuf_128 = (lid + 128u);
    float myVal_max_logitsBuf_128 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_128 = logitsBuf[neighborIdx_max_logitsBuf_128];
    bool isGreater_max_logitsBuf_128 = (neighborVal_max_logitsBuf_128 > myVal_max_logitsBuf_128);
    float maxVal_max_logitsBuf_128 = ((isGreater_max_logitsBuf_128) ? (neighborVal_max_logitsBuf_128) : (myVal_max_logitsBuf_128));
    logitsBuf[lid] = maxVal_max_logitsBuf_128;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_64 = (lid < 64u);
  if (shouldReduce_max_logitsBuf_64) {
    unsigned int neighborIdx_max_logitsBuf_64 = (lid + 64u);
    float myVal_max_logitsBuf_64 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_64 = logitsBuf[neighborIdx_max_logitsBuf_64];
    bool isGreater_max_logitsBuf_64 = (neighborVal_max_logitsBuf_64 > myVal_max_logitsBuf_64);
    float maxVal_max_logitsBuf_64 = ((isGreater_max_logitsBuf_64) ? (neighborVal_max_logitsBuf_64) : (myVal_max_logitsBuf_64));
    logitsBuf[lid] = maxVal_max_logitsBuf_64;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_32 = (lid < 32u);
  if (shouldReduce_max_logitsBuf_32) {
    unsigned int neighborIdx_max_logitsBuf_32 = (lid + 32u);
    float myVal_max_logitsBuf_32 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_32 = logitsBuf[neighborIdx_max_logitsBuf_32];
    bool isGreater_max_logitsBuf_32 = (neighborVal_max_logitsBuf_32 > myVal_max_logitsBuf_32);
    float maxVal_max_logitsBuf_32 = ((isGreater_max_logitsBuf_32) ? (neighborVal_max_logitsBuf_32) : (myVal_max_logitsBuf_32));
    logitsBuf[lid] = maxVal_max_logitsBuf_32;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_16 = (lid < 16u);
  if (shouldReduce_max_logitsBuf_16) {
    unsigned int neighborIdx_max_logitsBuf_16 = (lid + 16u);
    float myVal_max_logitsBuf_16 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_16 = logitsBuf[neighborIdx_max_logitsBuf_16];
    bool isGreater_max_logitsBuf_16 = (neighborVal_max_logitsBuf_16 > myVal_max_logitsBuf_16);
    float maxVal_max_logitsBuf_16 = ((isGreater_max_logitsBuf_16) ? (neighborVal_max_logitsBuf_16) : (myVal_max_logitsBuf_16));
    logitsBuf[lid] = maxVal_max_logitsBuf_16;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_8 = (lid < 8u);
  if (shouldReduce_max_logitsBuf_8) {
    unsigned int neighborIdx_max_logitsBuf_8 = (lid + 8u);
    float myVal_max_logitsBuf_8 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_8 = logitsBuf[neighborIdx_max_logitsBuf_8];
    bool isGreater_max_logitsBuf_8 = (neighborVal_max_logitsBuf_8 > myVal_max_logitsBuf_8);
    float maxVal_max_logitsBuf_8 = ((isGreater_max_logitsBuf_8) ? (neighborVal_max_logitsBuf_8) : (myVal_max_logitsBuf_8));
    logitsBuf[lid] = maxVal_max_logitsBuf_8;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_4 = (lid < 4u);
  if (shouldReduce_max_logitsBuf_4) {
    unsigned int neighborIdx_max_logitsBuf_4 = (lid + 4u);
    float myVal_max_logitsBuf_4 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_4 = logitsBuf[neighborIdx_max_logitsBuf_4];
    bool isGreater_max_logitsBuf_4 = (neighborVal_max_logitsBuf_4 > myVal_max_logitsBuf_4);
    float maxVal_max_logitsBuf_4 = ((isGreater_max_logitsBuf_4) ? (neighborVal_max_logitsBuf_4) : (myVal_max_logitsBuf_4));
    logitsBuf[lid] = maxVal_max_logitsBuf_4;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_2 = (lid < 2u);
  if (shouldReduce_max_logitsBuf_2) {
    unsigned int neighborIdx_max_logitsBuf_2 = (lid + 2u);
    float myVal_max_logitsBuf_2 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_2 = logitsBuf[neighborIdx_max_logitsBuf_2];
    bool isGreater_max_logitsBuf_2 = (neighborVal_max_logitsBuf_2 > myVal_max_logitsBuf_2);
    float maxVal_max_logitsBuf_2 = ((isGreater_max_logitsBuf_2) ? (neighborVal_max_logitsBuf_2) : (myVal_max_logitsBuf_2));
    logitsBuf[lid] = maxVal_max_logitsBuf_2;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_1 = (lid < 1u);
  if (shouldReduce_max_logitsBuf_1) {
    unsigned int neighborIdx_max_logitsBuf_1 = (lid + 1u);
    float myVal_max_logitsBuf_1 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_1 = logitsBuf[neighborIdx_max_logitsBuf_1];
    bool isGreater_max_logitsBuf_1 = (neighborVal_max_logitsBuf_1 > myVal_max_logitsBuf_1);
    float maxVal_max_logitsBuf_1 = ((isGreater_max_logitsBuf_1) ? (neighborVal_max_logitsBuf_1) : (myVal_max_logitsBuf_1));
    logitsBuf[lid] = maxVal_max_logitsBuf_1;
  }
  __syncthreads();
  float maxLogit = logitsBuf[0u];
  float originalLogit = ((valid) ? (routerLogits[globalIdx]) : (__uint_as_float(0xf149f2cau)));
  float shiftedLogit = (originalLogit - maxLogit);
  float expLogit = exp(shiftedLogit);
  float expMasked = ((expertValid) ? (expLogit) : (__uint_as_float(0x00000000u)));
  expBuf[lid] = expMasked;
  __syncthreads();
  bool shouldReduce_sum_expBuf_128 = (lid < 128u);
  if (shouldReduce_sum_expBuf_128) {
    unsigned int neighborIdx_sum_expBuf_128 = (lid + 128u);
    float myVal_sum_expBuf_128 = expBuf[lid];
    float neighborVal_sum_expBuf_128 = expBuf[neighborIdx_sum_expBuf_128];
    float sumVal_sum_expBuf_128 = (myVal_sum_expBuf_128 + neighborVal_sum_expBuf_128);
    expBuf[lid] = sumVal_sum_expBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_64 = (lid < 64u);
  if (shouldReduce_sum_expBuf_64) {
    unsigned int neighborIdx_sum_expBuf_64 = (lid + 64u);
    float myVal_sum_expBuf_64 = expBuf[lid];
    float neighborVal_sum_expBuf_64 = expBuf[neighborIdx_sum_expBuf_64];
    float sumVal_sum_expBuf_64 = (myVal_sum_expBuf_64 + neighborVal_sum_expBuf_64);
    expBuf[lid] = sumVal_sum_expBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_32 = (lid < 32u);
  if (shouldReduce_sum_expBuf_32) {
    unsigned int neighborIdx_sum_expBuf_32 = (lid + 32u);
    float myVal_sum_expBuf_32 = expBuf[lid];
    float neighborVal_sum_expBuf_32 = expBuf[neighborIdx_sum_expBuf_32];
    float sumVal_sum_expBuf_32 = (myVal_sum_expBuf_32 + neighborVal_sum_expBuf_32);
    expBuf[lid] = sumVal_sum_expBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_16 = (lid < 16u);
  if (shouldReduce_sum_expBuf_16) {
    unsigned int neighborIdx_sum_expBuf_16 = (lid + 16u);
    float myVal_sum_expBuf_16 = expBuf[lid];
    float neighborVal_sum_expBuf_16 = expBuf[neighborIdx_sum_expBuf_16];
    float sumVal_sum_expBuf_16 = (myVal_sum_expBuf_16 + neighborVal_sum_expBuf_16);
    expBuf[lid] = sumVal_sum_expBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_8 = (lid < 8u);
  if (shouldReduce_sum_expBuf_8) {
    unsigned int neighborIdx_sum_expBuf_8 = (lid + 8u);
    float myVal_sum_expBuf_8 = expBuf[lid];
    float neighborVal_sum_expBuf_8 = expBuf[neighborIdx_sum_expBuf_8];
    float sumVal_sum_expBuf_8 = (myVal_sum_expBuf_8 + neighborVal_sum_expBuf_8);
    expBuf[lid] = sumVal_sum_expBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_4 = (lid < 4u);
  if (shouldReduce_sum_expBuf_4) {
    unsigned int neighborIdx_sum_expBuf_4 = (lid + 4u);
    float myVal_sum_expBuf_4 = expBuf[lid];
    float neighborVal_sum_expBuf_4 = expBuf[neighborIdx_sum_expBuf_4];
    float sumVal_sum_expBuf_4 = (myVal_sum_expBuf_4 + neighborVal_sum_expBuf_4);
    expBuf[lid] = sumVal_sum_expBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_2 = (lid < 2u);
  if (shouldReduce_sum_expBuf_2) {
    unsigned int neighborIdx_sum_expBuf_2 = (lid + 2u);
    float myVal_sum_expBuf_2 = expBuf[lid];
    float neighborVal_sum_expBuf_2 = expBuf[neighborIdx_sum_expBuf_2];
    float sumVal_sum_expBuf_2 = (myVal_sum_expBuf_2 + neighborVal_sum_expBuf_2);
    expBuf[lid] = sumVal_sum_expBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_1 = (lid < 1u);
  if (shouldReduce_sum_expBuf_1) {
    unsigned int neighborIdx_sum_expBuf_1 = (lid + 1u);
    float myVal_sum_expBuf_1 = expBuf[lid];
    float neighborVal_sum_expBuf_1 = expBuf[neighborIdx_sum_expBuf_1];
    float sumVal_sum_expBuf_1 = (myVal_sum_expBuf_1 + neighborVal_sum_expBuf_1);
    expBuf[lid] = sumVal_sum_expBuf_1;
  }
  __syncthreads();
  float sumExp = expBuf[0u];
  float prob = (expLogit / sumExp);
  logitsBuf[lid] = prob;
  indexBuf[lid] = lidI32;
  __syncthreads();
  for (int kIter = 0; kIter < (int)(k); ++kIter) {
    if (lid == 0u) {
      int bestIdx = 0;
      float bestVal = __uint_as_float(0xf149f2cau);
      for (int scanIdx = 0; scanIdx < (int)(ne); ++scanIdx) {
        float scanVal = logitsBuf[scanIdx];
        bool isBetter = (scanVal > bestVal);
        if (isBetter) {
          bestVal = scanVal;
          int scanIdxI32 = ((int)(scanIdx));
          bestIdx = scanIdxI32;
        }
      }
      unsigned int kIterU32 = kIter;
      unsigned int outIdx = ((tokenIdx * k) + kIterU32);
      int bestIdxFinal = bestIdx;
      float bestValFinal = bestVal;
      expertIndices[outIdx] = bestIdxFinal;
      expertWeights[outIdx] = bestValFinal;
      unsigned int bestIdxU32 = ((unsigned int)(bestIdxFinal));
      logitsBuf[bestIdxU32] = __uint_as_float(0xf149f2cau);
    }
    __syncthreads();
  }
}
