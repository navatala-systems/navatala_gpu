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

kernel void navatala_transformer_softmax_with_mask_f16(device const half* _input [[buffer(0)]], device const uint* batchSize [[buffer(1)]], device const uint* seqLen [[buffer(2)]], device const bool* isCausal [[buffer(3)]], device half* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint rowIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  bool causal = isCausal[0u];
  threadgroup float maxBuf[256];
  threadgroup float sumBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  uint col = lid;
  uint row = (rowIdx % sl);
  uint globalIdx = ((rowIdx * sl) + lid);
  half rawXF16 = ((valid) ? (_input[globalIdx]) : (half(0.000000)));
  float rawX = ((colValid) ? (((float)(rawXF16))) : (as_type<float>(0xf149f2cau)));
  bool isMasked = (causal && (col > row));
  float x = ((isMasked) ? (as_type<float>(0xf149f2cau)) : (rawX));
  maxBuf[lid] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxVal = maxBuf[0u];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = (((colValid && (!isMasked))) ? (expVal) : (as_type<float>(0x00000000u)));
  sumBuf[lid] = expValMasked;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumExp = sumBuf[0u];
  if (valid) {
    float resultF32 = ((isMasked) ? (as_type<float>(0x00000000u)) : ((expVal / sumExp)));
    half result = ((half)(resultF32));
    _output[globalIdx] = result;
  }
}
