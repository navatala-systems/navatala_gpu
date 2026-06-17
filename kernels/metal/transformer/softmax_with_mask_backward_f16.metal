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

kernel void navatala_transformer_softmax_with_mask_backward_f16(device const half* _output [[buffer(0)]], device const half* gradOutput [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* seqLen [[buffer(3)]], device const bool* isCausal [[buffer(4)]], device half* gradInput [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint rowIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  bool causal = isCausal[0u];
  threadgroup float dotBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  uint col = lid;
  uint row = (rowIdx % sl);
  bool isMasked = (causal && (col > row));
  uint globalIdx = ((rowIdx * sl) + lid);
  half yF16 = ((valid) ? (_output[globalIdx]) : (half(0.000000)));
  half dyF16 = ((valid) ? (gradOutput[globalIdx]) : (half(0.000000)));
  float y = ((float)(yF16));
  float dy = ((float)(dyF16));
  float yDy = ((isMasked) ? (as_type<float>(0x00000000u)) : ((y * dy)));
  dotBuf[lid] = yDy;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_128 = (lid < 128u);
  if (shouldReduce_sum_dotBuf_128) {
    uint neighborIdx_sum_dotBuf_128 = (lid + 128u);
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_64 = (lid < 64u);
  if (shouldReduce_sum_dotBuf_64) {
    uint neighborIdx_sum_dotBuf_64 = (lid + 64u);
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_32 = (lid < 32u);
  if (shouldReduce_sum_dotBuf_32) {
    uint neighborIdx_sum_dotBuf_32 = (lid + 32u);
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_16 = (lid < 16u);
  if (shouldReduce_sum_dotBuf_16) {
    uint neighborIdx_sum_dotBuf_16 = (lid + 16u);
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_8 = (lid < 8u);
  if (shouldReduce_sum_dotBuf_8) {
    uint neighborIdx_sum_dotBuf_8 = (lid + 8u);
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_4 = (lid < 4u);
  if (shouldReduce_sum_dotBuf_4) {
    uint neighborIdx_sum_dotBuf_4 = (lid + 4u);
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_2 = (lid < 2u);
  if (shouldReduce_sum_dotBuf_2) {
    uint neighborIdx_sum_dotBuf_2 = (lid + 2u);
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool shouldReduce_sum_dotBuf_1 = (lid < 1u);
  if (shouldReduce_sum_dotBuf_1) {
    uint neighborIdx_sum_dotBuf_1 = (lid + 1u);
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float dot = dotBuf[0u];
  if (valid) {
    float dyMinusDot = (dy - dot);
    float dxF32Unmasked = (y * dyMinusDot);
    float dxF32 = ((isMasked) ? (as_type<float>(0x00000000u)) : (dxF32Unmasked));
    half dx = ((half)(dxF32));
    gradInput[globalIdx] = dx;
  }
}
