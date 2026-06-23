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
extern "C" __global__ void navatala_transformer_expert_capacity_mask_f16(const int* expertIndices, const unsigned int* numTokens, const unsigned int* numExperts, const unsigned int* topK, const unsigned int* capacity, __half* mask) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nt = numTokens[0u];
  unsigned int ne = numExperts[0u];
  unsigned int k = topK[0u];
  unsigned int cap = capacity[0u];
  __shared__ unsigned int expertCounts[256];
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  if (lid < ne) {
    expertCounts[lid] = 0u;
  }
  __syncthreads();
  unsigned int totalElements = (nt * k);
  bool valid = (gid < totalElements);
  if (valid) {
    int expertIdx = expertIndices[gid];
    unsigned int expertIdxU32 = ((unsigned int)(expertIdx));
    bool expertIdxValid = (expertIdxU32 < ne);
    if (expertIdxValid) {
      unsigned int prevCount = atomicAdd(&(expertCounts[expertIdxU32]), 1u);
      bool withinCapacity = (prevCount < cap);
      __half maskVal = ((withinCapacity) ? (__float2half(1.000000f)) : (__float2half(0.000000f)));
      mask[gid] = maskVal;
    } else {
      mask[gid] = __float2half(0.000000f);
    }
  }
}
