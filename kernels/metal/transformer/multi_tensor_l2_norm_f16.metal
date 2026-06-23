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

kernel void navatala_transformer_multi_tensor_l2_norm_f16(device const half* _input [[buffer(0)]], device const uint* totalElements [[buffer(1)]], device float* partialSums [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint gid = ((uint)(int(__tgid.x)));
  uint globalIdx = ((gid * 256u) + lid);
  uint total = totalElements[0u];
  threadgroup float sumSq[256];
  bool valid = (globalIdx < total);
  half xF16 = ((valid) ? (_input[globalIdx]) : (half(0.000000)));
  float x = ((float)(xF16));
  float xSq = (x * x);
  sumSq[lid] = xSq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_128 = (lid < 128u);
  if (shouldReduce_sumSq_128) {
    uint neighborIdx_sumSq_128 = (lid + 128u);
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_64 = (lid < 64u);
  if (shouldReduce_sumSq_64) {
    uint neighborIdx_sumSq_64 = (lid + 64u);
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_32 = (lid < 32u);
  if (shouldReduce_sumSq_32) {
    uint neighborIdx_sumSq_32 = (lid + 32u);
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_16 = (lid < 16u);
  if (shouldReduce_sumSq_16) {
    uint neighborIdx_sumSq_16 = (lid + 16u);
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_8 = (lid < 8u);
  if (shouldReduce_sumSq_8) {
    uint neighborIdx_sumSq_8 = (lid + 8u);
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_4 = (lid < 4u);
  if (shouldReduce_sumSq_4) {
    uint neighborIdx_sumSq_4 = (lid + 4u);
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_2 = (lid < 2u);
  if (shouldReduce_sumSq_2) {
    uint neighborIdx_sumSq_2 = (lid + 2u);
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sumSq_1 = (lid < 1u);
  if (shouldReduce_sumSq_1) {
    uint neighborIdx_sumSq_1 = (lid + 1u);
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    float partialSum = sumSq[0u];
    partialSums[gid] = partialSum;
  }
}
