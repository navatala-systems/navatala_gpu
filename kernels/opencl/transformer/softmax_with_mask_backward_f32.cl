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

__kernel void navatala_transformer_softmax_with_mask_backward_f32(__global const float* _output, __global const float* gradOutput, __global const uint* batchSize, __global const uint* seqLen, __global const bool* isCausal, __global float* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint rowIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  bool causal = isCausal[(uint)(0u)];
  __local float dotBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  uint col = lid;
  uint row = (rowIdx % sl);
  bool isMasked = (causal && (col > row));
  uint globalIdx = ((rowIdx * sl) + lid);
  float y = ((valid) ? (_output[globalIdx]) : (as_float(0x00000000u)));
  float dy = ((valid) ? (gradOutput[globalIdx]) : (as_float(0x00000000u)));
  float yDy = ((isMasked) ? (as_float(0x00000000u)) : ((y * dy)));
  dotBuf[lid] = yDy;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_dotBuf_128) {
    uint neighborIdx_sum_dotBuf_128 = (lid + (uint)(128u));
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_dotBuf_64) {
    uint neighborIdx_sum_dotBuf_64 = (lid + (uint)(64u));
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_dotBuf_32) {
    uint neighborIdx_sum_dotBuf_32 = (lid + (uint)(32u));
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_dotBuf_16) {
    uint neighborIdx_sum_dotBuf_16 = (lid + (uint)(16u));
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_dotBuf_8) {
    uint neighborIdx_sum_dotBuf_8 = (lid + (uint)(8u));
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_dotBuf_4) {
    uint neighborIdx_sum_dotBuf_4 = (lid + (uint)(4u));
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_dotBuf_2) {
    uint neighborIdx_sum_dotBuf_2 = (lid + (uint)(2u));
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_dotBuf_1) {
    uint neighborIdx_sum_dotBuf_1 = (lid + (uint)(1u));
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float dot = dotBuf[(uint)(0u)];
  if (valid) {
    float dyMinusDot = (dy - dot);
    float dxUnmasked = (y * dyMinusDot);
    float dx = ((isMasked) ? (as_float(0x00000000u)) : (dxUnmasked));
    gradInput[globalIdx] = dx;
  }
}
