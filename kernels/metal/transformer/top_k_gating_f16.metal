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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_top_k_gating_f16(device const half* routerLogits [[buffer(0)]], device const uint* numTokens [[buffer(1)]], device const uint* numExperts [[buffer(2)]], device const uint* topK [[buffer(3)]], device int* expertIndices [[buffer(4)]], device half* expertWeights [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint tokenIdx = ((uint)(int(__tgid.x)));
  uint nt = numTokens[0u];
  uint ne = numExperts[0u];
  uint k = topK[0u];
  threadgroup float logitsBuf[256];
  threadgroup float expBuf[256];
  threadgroup int indexBuf[256];
  bool tokenValid = (tokenIdx < nt);
  bool expertValid = (lid < ne);
  bool valid = (tokenValid && expertValid);
  uint globalIdx = ((tokenIdx * ne) + lid);
  half logitF16 = ((valid) ? (routerLogits[globalIdx]) : (half(0.000000)));
  float logit = ((expertValid) ? (((float)(logitF16))) : (as_type<float>(0xf149f2cau)));
  logitsBuf[lid] = logit;
  int lidI32 = ((int)(lid));
  indexBuf[lid] = lidI32;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_128 = (lid < 128u);
  if (shouldReduce_max_logitsBuf_128) {
    uint neighborIdx_max_logitsBuf_128 = (lid + 128u);
    float myVal_max_logitsBuf_128 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_128 = logitsBuf[neighborIdx_max_logitsBuf_128];
    bool isGreater_max_logitsBuf_128 = (neighborVal_max_logitsBuf_128 > myVal_max_logitsBuf_128);
    float maxVal_max_logitsBuf_128 = ((isGreater_max_logitsBuf_128) ? (neighborVal_max_logitsBuf_128) : (myVal_max_logitsBuf_128));
    logitsBuf[lid] = maxVal_max_logitsBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_64 = (lid < 64u);
  if (shouldReduce_max_logitsBuf_64) {
    uint neighborIdx_max_logitsBuf_64 = (lid + 64u);
    float myVal_max_logitsBuf_64 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_64 = logitsBuf[neighborIdx_max_logitsBuf_64];
    bool isGreater_max_logitsBuf_64 = (neighborVal_max_logitsBuf_64 > myVal_max_logitsBuf_64);
    float maxVal_max_logitsBuf_64 = ((isGreater_max_logitsBuf_64) ? (neighborVal_max_logitsBuf_64) : (myVal_max_logitsBuf_64));
    logitsBuf[lid] = maxVal_max_logitsBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_32 = (lid < 32u);
  if (shouldReduce_max_logitsBuf_32) {
    uint neighborIdx_max_logitsBuf_32 = (lid + 32u);
    float myVal_max_logitsBuf_32 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_32 = logitsBuf[neighborIdx_max_logitsBuf_32];
    bool isGreater_max_logitsBuf_32 = (neighborVal_max_logitsBuf_32 > myVal_max_logitsBuf_32);
    float maxVal_max_logitsBuf_32 = ((isGreater_max_logitsBuf_32) ? (neighborVal_max_logitsBuf_32) : (myVal_max_logitsBuf_32));
    logitsBuf[lid] = maxVal_max_logitsBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_16 = (lid < 16u);
  if (shouldReduce_max_logitsBuf_16) {
    uint neighborIdx_max_logitsBuf_16 = (lid + 16u);
    float myVal_max_logitsBuf_16 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_16 = logitsBuf[neighborIdx_max_logitsBuf_16];
    bool isGreater_max_logitsBuf_16 = (neighborVal_max_logitsBuf_16 > myVal_max_logitsBuf_16);
    float maxVal_max_logitsBuf_16 = ((isGreater_max_logitsBuf_16) ? (neighborVal_max_logitsBuf_16) : (myVal_max_logitsBuf_16));
    logitsBuf[lid] = maxVal_max_logitsBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_8 = (lid < 8u);
  if (shouldReduce_max_logitsBuf_8) {
    uint neighborIdx_max_logitsBuf_8 = (lid + 8u);
    float myVal_max_logitsBuf_8 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_8 = logitsBuf[neighborIdx_max_logitsBuf_8];
    bool isGreater_max_logitsBuf_8 = (neighborVal_max_logitsBuf_8 > myVal_max_logitsBuf_8);
    float maxVal_max_logitsBuf_8 = ((isGreater_max_logitsBuf_8) ? (neighborVal_max_logitsBuf_8) : (myVal_max_logitsBuf_8));
    logitsBuf[lid] = maxVal_max_logitsBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_4 = (lid < 4u);
  if (shouldReduce_max_logitsBuf_4) {
    uint neighborIdx_max_logitsBuf_4 = (lid + 4u);
    float myVal_max_logitsBuf_4 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_4 = logitsBuf[neighborIdx_max_logitsBuf_4];
    bool isGreater_max_logitsBuf_4 = (neighborVal_max_logitsBuf_4 > myVal_max_logitsBuf_4);
    float maxVal_max_logitsBuf_4 = ((isGreater_max_logitsBuf_4) ? (neighborVal_max_logitsBuf_4) : (myVal_max_logitsBuf_4));
    logitsBuf[lid] = maxVal_max_logitsBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_2 = (lid < 2u);
  if (shouldReduce_max_logitsBuf_2) {
    uint neighborIdx_max_logitsBuf_2 = (lid + 2u);
    float myVal_max_logitsBuf_2 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_2 = logitsBuf[neighborIdx_max_logitsBuf_2];
    bool isGreater_max_logitsBuf_2 = (neighborVal_max_logitsBuf_2 > myVal_max_logitsBuf_2);
    float maxVal_max_logitsBuf_2 = ((isGreater_max_logitsBuf_2) ? (neighborVal_max_logitsBuf_2) : (myVal_max_logitsBuf_2));
    logitsBuf[lid] = maxVal_max_logitsBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_logitsBuf_1 = (lid < 1u);
  if (shouldReduce_max_logitsBuf_1) {
    uint neighborIdx_max_logitsBuf_1 = (lid + 1u);
    float myVal_max_logitsBuf_1 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_1 = logitsBuf[neighborIdx_max_logitsBuf_1];
    bool isGreater_max_logitsBuf_1 = (neighborVal_max_logitsBuf_1 > myVal_max_logitsBuf_1);
    float maxVal_max_logitsBuf_1 = ((isGreater_max_logitsBuf_1) ? (neighborVal_max_logitsBuf_1) : (myVal_max_logitsBuf_1));
    logitsBuf[lid] = maxVal_max_logitsBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxLogit = logitsBuf[0u];
  half originalLogitF16 = ((valid) ? (routerLogits[globalIdx]) : (half(0.000000)));
  float originalLogit = ((expertValid) ? (((float)(originalLogitF16))) : (as_type<float>(0xf149f2cau)));
  float shiftedLogit = (originalLogit - maxLogit);
  float expLogit = exp(shiftedLogit);
  float expMasked = ((expertValid) ? (expLogit) : (as_type<float>(0x00000000u)));
  expBuf[lid] = expMasked;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_128 = (lid < 128u);
  if (shouldReduce_sum_expBuf_128) {
    uint neighborIdx_sum_expBuf_128 = (lid + 128u);
    float myVal_sum_expBuf_128 = expBuf[lid];
    float neighborVal_sum_expBuf_128 = expBuf[neighborIdx_sum_expBuf_128];
    float sumVal_sum_expBuf_128 = (myVal_sum_expBuf_128 + neighborVal_sum_expBuf_128);
    expBuf[lid] = sumVal_sum_expBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_64 = (lid < 64u);
  if (shouldReduce_sum_expBuf_64) {
    uint neighborIdx_sum_expBuf_64 = (lid + 64u);
    float myVal_sum_expBuf_64 = expBuf[lid];
    float neighborVal_sum_expBuf_64 = expBuf[neighborIdx_sum_expBuf_64];
    float sumVal_sum_expBuf_64 = (myVal_sum_expBuf_64 + neighborVal_sum_expBuf_64);
    expBuf[lid] = sumVal_sum_expBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_32 = (lid < 32u);
  if (shouldReduce_sum_expBuf_32) {
    uint neighborIdx_sum_expBuf_32 = (lid + 32u);
    float myVal_sum_expBuf_32 = expBuf[lid];
    float neighborVal_sum_expBuf_32 = expBuf[neighborIdx_sum_expBuf_32];
    float sumVal_sum_expBuf_32 = (myVal_sum_expBuf_32 + neighborVal_sum_expBuf_32);
    expBuf[lid] = sumVal_sum_expBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_16 = (lid < 16u);
  if (shouldReduce_sum_expBuf_16) {
    uint neighborIdx_sum_expBuf_16 = (lid + 16u);
    float myVal_sum_expBuf_16 = expBuf[lid];
    float neighborVal_sum_expBuf_16 = expBuf[neighborIdx_sum_expBuf_16];
    float sumVal_sum_expBuf_16 = (myVal_sum_expBuf_16 + neighborVal_sum_expBuf_16);
    expBuf[lid] = sumVal_sum_expBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_8 = (lid < 8u);
  if (shouldReduce_sum_expBuf_8) {
    uint neighborIdx_sum_expBuf_8 = (lid + 8u);
    float myVal_sum_expBuf_8 = expBuf[lid];
    float neighborVal_sum_expBuf_8 = expBuf[neighborIdx_sum_expBuf_8];
    float sumVal_sum_expBuf_8 = (myVal_sum_expBuf_8 + neighborVal_sum_expBuf_8);
    expBuf[lid] = sumVal_sum_expBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_4 = (lid < 4u);
  if (shouldReduce_sum_expBuf_4) {
    uint neighborIdx_sum_expBuf_4 = (lid + 4u);
    float myVal_sum_expBuf_4 = expBuf[lid];
    float neighborVal_sum_expBuf_4 = expBuf[neighborIdx_sum_expBuf_4];
    float sumVal_sum_expBuf_4 = (myVal_sum_expBuf_4 + neighborVal_sum_expBuf_4);
    expBuf[lid] = sumVal_sum_expBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_2 = (lid < 2u);
  if (shouldReduce_sum_expBuf_2) {
    uint neighborIdx_sum_expBuf_2 = (lid + 2u);
    float myVal_sum_expBuf_2 = expBuf[lid];
    float neighborVal_sum_expBuf_2 = expBuf[neighborIdx_sum_expBuf_2];
    float sumVal_sum_expBuf_2 = (myVal_sum_expBuf_2 + neighborVal_sum_expBuf_2);
    expBuf[lid] = sumVal_sum_expBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_expBuf_1 = (lid < 1u);
  if (shouldReduce_sum_expBuf_1) {
    uint neighborIdx_sum_expBuf_1 = (lid + 1u);
    float myVal_sum_expBuf_1 = expBuf[lid];
    float neighborVal_sum_expBuf_1 = expBuf[neighborIdx_sum_expBuf_1];
    float sumVal_sum_expBuf_1 = (myVal_sum_expBuf_1 + neighborVal_sum_expBuf_1);
    expBuf[lid] = sumVal_sum_expBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = expBuf[0u];
  float prob = (expLogit / sumExp);
  logitsBuf[lid] = prob;
  int lidI32_2 = ((int)(lid));
  indexBuf[lid] = lidI32_2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int kIter = 0; kIter < (int)(k); ++kIter) {
    if ((lid == 0u)) {
      int bestIdx = 0;
      float bestVal = as_type<float>(0xf149f2cau);
      for (int scanIdx = 0; scanIdx < (int)(ne); ++scanIdx) {
        float scanVal = logitsBuf[scanIdx];
        bool isBetter = (scanVal > bestVal);
        if (isBetter) {
          bestVal = scanVal;
          int scanIdxI32 = ((int)(scanIdx));
          bestIdx = scanIdxI32;
        }
      }
      uint kIterU32 = kIter;
      uint outIdx = ((tokenIdx * k) + kIterU32);
      int bestIdxFinal = bestIdx;
      float bestValFinal = bestVal;
      half bestValF16 = ((half)(bestValFinal));
      expertIndices[outIdx] = bestIdxFinal;
      expertWeights[outIdx] = bestValF16;
      uint bestIdxU32 = ((uint)(bestIdxFinal));
      logitsBuf[bestIdxU32] = as_type<float>(0xf149f2cau);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
