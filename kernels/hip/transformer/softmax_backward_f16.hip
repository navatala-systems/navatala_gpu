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

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_softmax_backward_f16(const __half* _output, const __half* gradOutput, const unsigned int* batchSize, const unsigned int* seqLen, __half* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  __shared__ float dotBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  unsigned int globalIdx = ((batchIdx * sl) + lid);
  __half yF16 = (((batchValid && seqValid)) ? (_output[globalIdx]) : (__float2half(0.000000f)));
  __half dyF16 = (((batchValid && seqValid)) ? (gradOutput[globalIdx]) : (__float2half(0.000000f)));
  float y = ((float)(yF16));
  float dy = ((float)(dyF16));
  float yDy = (y * dy);
  dotBuf[lid] = yDy;
  __syncthreads();
  bool shouldReduce_sum_dotBuf_128 = (lid < 128u);
  if (shouldReduce_sum_dotBuf_128) {
    unsigned int neighborIdx_sum_dotBuf_128 = (lid + 128u);
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_64 = (lid < 64u);
  if (shouldReduce_sum_dotBuf_64) {
    unsigned int neighborIdx_sum_dotBuf_64 = (lid + 64u);
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_32 = (lid < 32u);
  if (shouldReduce_sum_dotBuf_32) {
    unsigned int neighborIdx_sum_dotBuf_32 = (lid + 32u);
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_16 = (lid < 16u);
  if (shouldReduce_sum_dotBuf_16) {
    unsigned int neighborIdx_sum_dotBuf_16 = (lid + 16u);
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_8 = (lid < 8u);
  if (shouldReduce_sum_dotBuf_8) {
    unsigned int neighborIdx_sum_dotBuf_8 = (lid + 8u);
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_4 = (lid < 4u);
  if (shouldReduce_sum_dotBuf_4) {
    unsigned int neighborIdx_sum_dotBuf_4 = (lid + 4u);
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_2 = (lid < 2u);
  if (shouldReduce_sum_dotBuf_2) {
    unsigned int neighborIdx_sum_dotBuf_2 = (lid + 2u);
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_1 = (lid < 1u);
  if (shouldReduce_sum_dotBuf_1) {
    unsigned int neighborIdx_sum_dotBuf_1 = (lid + 1u);
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  __syncthreads();
  float dot = dotBuf[0u];
  if (batchValid && seqValid) {
    float dyMinusDot = (dy - dot);
    float dxF32 = (y * dyMinusDot);
    __half dx = ((__half)(dxF32));
    gradInput[globalIdx] = dx;
  }
}
