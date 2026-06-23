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
extern "C" __global__ void navatala_transformer_permute_tokens_f16(const __half* hiddenStates, const int* expertIndices, const unsigned int* expertOffsets, unsigned int* expertCounters, const unsigned int* numTokens, const unsigned int* hiddenSize, const unsigned int* numExperts, __half* permutedStates, int* tokenIndices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tokenIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nt = numTokens[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int ne = numExperts[0u];
  __shared__ unsigned int sharedDestTokenIdx[1];
  bool tokenValid = (tokenIdx < nt);
  if (tokenValid) {
    int expertIdx = expertIndices[tokenIdx];
    unsigned int expertIdxU32 = ((unsigned int)(expertIdx));
    bool expertValid = (expertIdxU32 < ne);
    if (expertValid) {
      if (lid == 0u) {
        unsigned int localPosition = atomicAdd(&(expertCounters[expertIdxU32]), 1u);
        unsigned int expertOffset = expertOffsets[expertIdxU32];
        unsigned int destTokenIdx = (expertOffset + localPosition);
        sharedDestTokenIdx[0u] = destTokenIdx;
        int tokenIdxI32 = ((int)(tokenIdx));
        tokenIndices[destTokenIdx] = tokenIdxI32;
      }
      __syncthreads();
      unsigned int destTokenIdx = sharedDestTokenIdx[0u];
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        unsigned int hIdx = (lid + (256u * hIter));
        if (hIdx < hs) {
          unsigned int srcIdx = ((tokenIdx * hs) + hIdx);
          unsigned int dstIdx = ((destTokenIdx * hs) + hIdx);
          __half val = hiddenStates[srcIdx];
          permutedStates[dstIdx] = val;
        }
      }
    }
  }
}
