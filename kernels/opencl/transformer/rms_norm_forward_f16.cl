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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_rms_norm_forward_f16(__global const half* _input, __global const half* gamma, __global const float* epsilon, __global const uint* batchSize, __global const uint* hiddenSize, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  float eps = epsilon[(uint)(0u)];
  __local float sumSq[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : ((half)(0.000000f)));
  float x = ((float)(xF16));
  float xSq = (x * x);
  sumSq[lid] = ((hiddenValid) ? (xSq) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_128 = (lid < (uint)(128u));
  if (shouldReduce_sumSq_128) {
    uint neighborIdx_sumSq_128 = (lid + (uint)(128u));
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_64 = (lid < (uint)(64u));
  if (shouldReduce_sumSq_64) {
    uint neighborIdx_sumSq_64 = (lid + (uint)(64u));
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_32 = (lid < (uint)(32u));
  if (shouldReduce_sumSq_32) {
    uint neighborIdx_sumSq_32 = (lid + (uint)(32u));
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_16 = (lid < (uint)(16u));
  if (shouldReduce_sumSq_16) {
    uint neighborIdx_sumSq_16 = (lid + (uint)(16u));
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_8 = (lid < (uint)(8u));
  if (shouldReduce_sumSq_8) {
    uint neighborIdx_sumSq_8 = (lid + (uint)(8u));
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_4 = (lid < (uint)(4u));
  if (shouldReduce_sumSq_4) {
    uint neighborIdx_sumSq_4 = (lid + (uint)(4u));
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_2 = (lid < (uint)(2u));
  if (shouldReduce_sumSq_2) {
    uint neighborIdx_sumSq_2 = (lid + (uint)(2u));
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_1 = (lid < (uint)(1u));
  if (shouldReduce_sumSq_1) {
    uint neighborIdx_sumSq_1 = (lid + (uint)(1u));
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSumSq = sumSq[(uint)(0u)];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  if ((batchValid && hiddenValid)) {
    half gF16 = gamma[lid];
    float g = ((float)(gF16));
    float xNorm = (x / rms);
    float resultF32 = (g * xNorm);
    half result = ((half)(resultF32));
    _output[globalIdx] = result;
  }
}
