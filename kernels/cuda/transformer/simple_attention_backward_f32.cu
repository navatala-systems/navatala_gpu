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
extern "C" __global__ void navatala_transformer_simple_attention_backward_f32(const float* gradOutput, const float* query, const float* key, const float* value, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* numHeads, const unsigned int* headDim, float* gradQuery, float* gradKey, float* gradValue) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int seqHeads = (sl * nh);
  unsigned int batchIdx = (groupId / seqHeads);
  unsigned int remainder = (groupId % seqHeads);
  unsigned int queryPos = (remainder / nh);
  unsigned int headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (__uint_as_float(0x3f800000u) / sqrtHd);
  __shared__ float attnScores[256];
  __shared__ float sumBuf[256];
  __shared__ float gradBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  unsigned int headDimStride = hd;
  unsigned int headStride = (nh * hd);
  unsigned int seqStride = (sl * headStride);
  unsigned int qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  unsigned int kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProdQK = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      float qVal = query[qIdx];
      float kVal = key[kIdx];
      float prod = (qVal * kVal);
      float oldDotProdQK = dotProdQK;
      dotProdQK = (oldDotProdQK + prod);
    }
    float finalDotProdQK = dotProdQK;
    float scaledScore = (finalDotProdQK * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = __uint_as_float(0xf149f2cau);
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    unsigned int neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    unsigned int neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    unsigned int neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    unsigned int neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    unsigned int neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    unsigned int neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    unsigned int neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    unsigned int neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  __syncthreads();
  float maxScore = attnScores[0u];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    unsigned int neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    unsigned int neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    unsigned int neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    unsigned int neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    unsigned int neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    unsigned int neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    unsigned int neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    unsigned int neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    unsigned int vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float dotProdGradAttn = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int gradOutIdx = (qBase + ((unsigned int)(d)));
      unsigned int vIdx = (vBase + ((unsigned int)(d)));
      float gradOutVal = gradOutput[gradOutIdx];
      float vVal = value[vIdx];
      float prodGradV = (gradOutVal * vVal);
      float oldDotProdGradAttn = dotProdGradAttn;
      dotProdGradAttn = (oldDotProdGradAttn + prodGradV);
      float gradVContrib = (attnWeight * gradOutVal);
      float oldGradV = gradValue[vIdx];
      float newGradV = (oldGradV + gradVContrib);
      gradValue[vIdx] = newGradV;
    }
    float finalGradAttn = dotProdGradAttn;
    gradBuf[lid] = finalGradAttn;
  } else {
    gradBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  if (valid) {
    float myAttnWeight = (exp((attnScores[lid] - maxScore)) / sumExp);
    float myGradAttn = gradBuf[lid];
    float gradScore = (myAttnWeight * myGradAttn);
    float scaledGradScore = (gradScore * scale);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      float kVal = key[kIdx];
      float qVal = query[qIdx];
      float gradQContrib = (scaledGradScore * kVal);
      float oldGradQ = gradQuery[qIdx];
      float newGradQ = (oldGradQ + gradQContrib);
      gradQuery[qIdx] = newGradQ;
      float gradKContrib = (scaledGradScore * qVal);
      float oldGradK = gradKey[kIdx];
      float newGradK = (oldGradK + gradKContrib);
      gradKey[kIdx] = newGradK;
    }
  }
}
