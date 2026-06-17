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
extern "C" __global__ void navatala_transformer_rms_norm_forward_f32(const float* _input, const float* gamma, const float* epsilon, const unsigned int* batchSize, const unsigned int* hiddenSize, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  float eps = epsilon[0u];
  __shared__ float sumSq[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  unsigned int globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (__uint_as_float(0x00000000u)));
  float xSq = (x * x);
  sumSq[lid] = ((hiddenValid) ? (xSq) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumSq_128 = (lid < 128u);
  if (shouldReduce_sumSq_128) {
    unsigned int neighborIdx_sumSq_128 = (lid + 128u);
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  __syncthreads();
  bool shouldReduce_sumSq_64 = (lid < 64u);
  if (shouldReduce_sumSq_64) {
    unsigned int neighborIdx_sumSq_64 = (lid + 64u);
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  __syncthreads();
  bool shouldReduce_sumSq_32 = (lid < 32u);
  if (shouldReduce_sumSq_32) {
    unsigned int neighborIdx_sumSq_32 = (lid + 32u);
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  __syncthreads();
  bool shouldReduce_sumSq_16 = (lid < 16u);
  if (shouldReduce_sumSq_16) {
    unsigned int neighborIdx_sumSq_16 = (lid + 16u);
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  __syncthreads();
  bool shouldReduce_sumSq_8 = (lid < 8u);
  if (shouldReduce_sumSq_8) {
    unsigned int neighborIdx_sumSq_8 = (lid + 8u);
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  __syncthreads();
  bool shouldReduce_sumSq_4 = (lid < 4u);
  if (shouldReduce_sumSq_4) {
    unsigned int neighborIdx_sumSq_4 = (lid + 4u);
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  __syncthreads();
  bool shouldReduce_sumSq_2 = (lid < 2u);
  if (shouldReduce_sumSq_2) {
    unsigned int neighborIdx_sumSq_2 = (lid + 2u);
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  __syncthreads();
  bool shouldReduce_sumSq_1 = (lid < 1u);
  if (shouldReduce_sumSq_1) {
    unsigned int neighborIdx_sumSq_1 = (lid + 1u);
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  __syncthreads();
  float totalSumSq = sumSq[0u];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  if ((batchValid && hiddenValid)) {
    float g = gamma[lid];
    float xNorm = (x / rms);
    float result = (g * xNorm);
    _output[globalIdx] = result;
  }
}
