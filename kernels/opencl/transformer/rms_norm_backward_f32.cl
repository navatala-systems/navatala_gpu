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

__kernel void navatala_transformer_rms_norm_backward_f32(__global const float* _input, __global const float* gradOutput, __global const float* gamma, __global const float* rsigma, __global const uint* batchSize, __global const uint* hiddenSize, __global float* gradInput, __global float* gradGamma) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  __local float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (as_float(0x00000000u)));
  float dy = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : (as_float(0x00000000u)));
  float g = ((hiddenValid) ? (gamma[lid]) : (as_float(0x00000000u)));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (as_float(0x3f800000u)));
  float xHat = (x * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_128 = (lid < (uint)(128u));
  if (shouldReduce_sumDyGammaXhat_128) {
    uint neighborIdx_sumDyGammaXhat_128 = (lid + (uint)(128u));
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_64 = (lid < (uint)(64u));
  if (shouldReduce_sumDyGammaXhat_64) {
    uint neighborIdx_sumDyGammaXhat_64 = (lid + (uint)(64u));
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_32 = (lid < (uint)(32u));
  if (shouldReduce_sumDyGammaXhat_32) {
    uint neighborIdx_sumDyGammaXhat_32 = (lid + (uint)(32u));
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_16 = (lid < (uint)(16u));
  if (shouldReduce_sumDyGammaXhat_16) {
    uint neighborIdx_sumDyGammaXhat_16 = (lid + (uint)(16u));
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_8 = (lid < (uint)(8u));
  if (shouldReduce_sumDyGammaXhat_8) {
    uint neighborIdx_sumDyGammaXhat_8 = (lid + (uint)(8u));
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_4 = (lid < (uint)(4u));
  if (shouldReduce_sumDyGammaXhat_4) {
    uint neighborIdx_sumDyGammaXhat_4 = (lid + (uint)(4u));
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_2 = (lid < (uint)(2u));
  if (shouldReduce_sumDyGammaXhat_2) {
    uint neighborIdx_sumDyGammaXhat_2 = (lid + (uint)(2u));
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_1 = (lid < (uint)(1u));
  if (shouldReduce_sumDyGammaXhat_1) {
    uint neighborIdx_sumDyGammaXhat_1 = (lid + (uint)(1u));
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalDyGammaXhat = sumDyGammaXhat[(uint)(0u)];
  float hsF32 = ((float)(hs));
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if (batchValid && hiddenValid) {
    float term1 = dyGamma;
    float term2 = (xHat * meanDyGammaXhat);
    float inner = (term1 - term2);
    float dx = (rs * inner);
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomic_add(&gradGamma[lid], dyXhat);
  }
}
