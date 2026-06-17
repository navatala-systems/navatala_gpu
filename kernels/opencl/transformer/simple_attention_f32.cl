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

__kernel void navatala_transformer_simple_attention_f32(__global const float* query, __global const float* key, __global const float* value, __global const uint* batchSize, __global const uint* seqLen, __global const uint* numHeads, __global const uint* headDim, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_float(0x3f800000u) / sqrtHd);
  __local float attnScores[256];
  __local float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProd = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      float qVal = query[qIdx];
      float kVal = key[kIdx];
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = as_float(0xf149f2cau);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_128 = (lid < (uint)(128u));
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + (uint)(128u));
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_64 = (lid < (uint)(64u));
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + (uint)(64u));
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_32 = (lid < (uint)(32u));
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + (uint)(32u));
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_16 = (lid < (uint)(16u));
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + (uint)(16u));
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_8 = (lid < (uint)(8u));
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + (uint)(8u));
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_4 = (lid < (uint)(4u));
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + (uint)(4u));
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_2 = (lid < (uint)(2u));
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + (uint)(2u));
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_1 = (lid < (uint)(1u));
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + (uint)(1u));
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxScore = attnScores[(uint)(0u)];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float v0 = value[vBase];
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((batchValid && (lid == (uint)(0u)))) {
    float outVal = sumBuf[(uint)(0u)];
    uint outIdx = (qBase + (uint)(0u));
    _output[outIdx] = outVal;
  }
}
