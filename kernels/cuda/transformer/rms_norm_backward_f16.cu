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
#include <cuda_fp16.h>
extern "C" __global__ void navatala_transformer_rms_norm_backward_f16(const __half* _input, const __half* gradOutput, const __half* gamma, const float* rsigma, const unsigned int* batchSize, const unsigned int* hiddenSize, __half* gradInput, float* gradGamma) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  __shared__ float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  unsigned int globalIdx = ((batchIdx * hs) + lid);
  __half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (__float2half_rn(0.000000f)));
  float x = ((float)(xF16));
  __half dyF16 = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : (__float2half_rn(0.000000f)));
  float dy = ((float)(dyF16));
  __half gF16 = ((hiddenValid) ? (gamma[lid]) : (__float2half_rn(0.000000f)));
  float g = ((float)(gF16));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (__uint_as_float(0x3f800000u)));
  float xHat = (x * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_128 = (lid < 128u);
  if (shouldReduce_sumDyGammaXhat_128) {
    unsigned int neighborIdx_sumDyGammaXhat_128 = (lid + 128u);
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_64 = (lid < 64u);
  if (shouldReduce_sumDyGammaXhat_64) {
    unsigned int neighborIdx_sumDyGammaXhat_64 = (lid + 64u);
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_32 = (lid < 32u);
  if (shouldReduce_sumDyGammaXhat_32) {
    unsigned int neighborIdx_sumDyGammaXhat_32 = (lid + 32u);
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_16 = (lid < 16u);
  if (shouldReduce_sumDyGammaXhat_16) {
    unsigned int neighborIdx_sumDyGammaXhat_16 = (lid + 16u);
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_8 = (lid < 8u);
  if (shouldReduce_sumDyGammaXhat_8) {
    unsigned int neighborIdx_sumDyGammaXhat_8 = (lid + 8u);
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_4 = (lid < 4u);
  if (shouldReduce_sumDyGammaXhat_4) {
    unsigned int neighborIdx_sumDyGammaXhat_4 = (lid + 4u);
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_2 = (lid < 2u);
  if (shouldReduce_sumDyGammaXhat_2) {
    unsigned int neighborIdx_sumDyGammaXhat_2 = (lid + 2u);
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_1 = (lid < 1u);
  if (shouldReduce_sumDyGammaXhat_1) {
    unsigned int neighborIdx_sumDyGammaXhat_1 = (lid + 1u);
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  __syncthreads();
  float totalDyGammaXhat = sumDyGammaXhat[0u];
  float hsF32 = ((float)(hs));
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if ((batchValid && hiddenValid)) {
    float term1 = dyGamma;
    float term2 = (xHat * meanDyGammaXhat);
    float inner = (term1 - term2);
    float dxF32 = (rs * inner);
    __half dx = ((__half)(dxF32));
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomicAdd(&gradGamma[lid], dyXhat);
  }
}
