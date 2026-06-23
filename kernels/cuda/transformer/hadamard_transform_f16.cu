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
extern "C" __global__ void navatala_transformer_hadamard_transform_f16(const __half* _input, const unsigned int* batchSize, const unsigned int* vectorSize, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int vs = vectorSize[0u];
  __shared__ float sharedBuf[256];
  bool batchValid = (batchIdx < bs);
  bool threadValid = (lid < vs);
  unsigned int globalIdx = ((batchIdx * vs) + lid);
  if (batchValid && threadValid) {
    __half valF16 = _input[globalIdx];
    float val = ((float)(valF16));
    sharedBuf[lid] = val;
  } else {
    sharedBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int stride1 = 1u;
  unsigned int partner1 = (lid ^ stride1);
  float myVal1 = sharedBuf[lid];
  float partnerVal1 = sharedBuf[partner1];
  bool lowBit1 = ((lid & stride1) == 0u);
  float newVal1 = ((lowBit1) ? ((myVal1 + partnerVal1)) : ((partnerVal1 - myVal1)));
  __syncthreads();
  sharedBuf[lid] = newVal1;
  __syncthreads();
  unsigned int stride2 = 2u;
  unsigned int partner2 = (lid ^ stride2);
  float myVal2 = sharedBuf[lid];
  float partnerVal2 = sharedBuf[partner2];
  bool lowBit2 = ((lid & stride2) == 0u);
  float newVal2 = ((lowBit2) ? ((myVal2 + partnerVal2)) : ((partnerVal2 - myVal2)));
  __syncthreads();
  sharedBuf[lid] = newVal2;
  __syncthreads();
  unsigned int stride3 = 4u;
  unsigned int partner3 = (lid ^ stride3);
  float myVal3 = sharedBuf[lid];
  float partnerVal3 = sharedBuf[partner3];
  bool lowBit3 = ((lid & stride3) == 0u);
  float newVal3 = ((lowBit3) ? ((myVal3 + partnerVal3)) : ((partnerVal3 - myVal3)));
  __syncthreads();
  sharedBuf[lid] = newVal3;
  __syncthreads();
  unsigned int stride4 = 8u;
  unsigned int partner4 = (lid ^ stride4);
  float myVal4 = sharedBuf[lid];
  float partnerVal4 = sharedBuf[partner4];
  bool lowBit4 = ((lid & stride4) == 0u);
  float newVal4 = ((lowBit4) ? ((myVal4 + partnerVal4)) : ((partnerVal4 - myVal4)));
  float vsF32 = ((float)(vs));
  float scale = sqrt(vsF32);
  float normalized = (newVal4 / scale);
  __half resultF16 = ((__half)(normalized));
  if (batchValid && threadValid) {
    _output[globalIdx] = resultF16;
  }
}
