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

__kernel void navatala_transformer_softmax_multi_pass_f32(__global const float* _input, __global const uint* batchSize, __global const uint* seqLen, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  __local float maxBuf[256];
  __local float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * sl);
  uint workgroupSize = (uint)(256u);
  float partialMax = as_float(0xf149f2cau);
  float partialSum = as_float(0x00000000u);
  uint iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!(iterIdx < sl)) break;
    uint globalIdx = (baseIdx + iterIdx);
    float xVal = ((batchValid) ? (_input[globalIdx]) : (as_float(0xf149f2cau)));
    bool isGreater = (xVal > partialMax);
    partialMax = ((isGreater) ? (xVal) : (partialMax));
    iterIdx = (iterIdx + workgroupSize);
  }
  maxBuf[lid] = partialMax;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + (uint)(128u));
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + (uint)(64u));
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + (uint)(32u));
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + (uint)(16u));
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + (uint)(8u));
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + (uint)(4u));
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + (uint)(2u));
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + (uint)(1u));
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float globalMax = maxBuf[(uint)(0u)];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!(iterIdx < sl)) break;
    if (batchValid) {
      uint globalIdx2 = (baseIdx + iterIdx);
      float xVal2 = _input[globalIdx2];
      float xShifted = (xVal2 - globalMax);
      float expVal = exp(xShifted);
      partialSum = (partialSum + expVal);
    }
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float globalSum = sumBuf[(uint)(0u)];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!(iterIdx < sl)) break;
    if (batchValid) {
      uint globalIdx3 = (baseIdx + iterIdx);
      float xVal3 = _input[globalIdx3];
      float xShifted3 = (xVal3 - globalMax);
      float expVal3 = exp(xShifted3);
      float result = (expVal3 / globalSum);
      _output[globalIdx3] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}
