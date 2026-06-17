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
extern "C" __global__ void navatala_transformer_kv_cache_append_f16(const __half* newK, const __half* newV, const unsigned int* blockTables, const unsigned int* seqPositions, const unsigned int* numHeads, const unsigned int* headDim, const unsigned int* pageSize, const unsigned int* maxBlocks, __half* kvCache) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int ps = pageSize[0u];
  unsigned int mb = maxBlocks[0u];
  unsigned int batchIdx = (groupId / nh);
  unsigned int headIdx = (groupId % nh);
  unsigned int seqPos = seqPositions[batchIdx];
  unsigned int blockIdx = (seqPos / ps);
  unsigned int blockOffset = (seqPos % ps);
  unsigned int blockTableBase = (batchIdx * mb);
  unsigned int physicalBlock = blockTables[(blockTableBase + blockIdx)];
  unsigned int kvBlockStride = (2u * (nh * (ps * hd)));
  unsigned int kvSlotStride = (nh * (ps * hd));
  unsigned int kvHeadStride = (ps * hd);
  unsigned int kCacheBase = ((physicalBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (blockOffset * hd)));
  unsigned int vCacheBase = ((physicalBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (blockOffset * hd))));
  unsigned int srcBase = ((batchIdx * (nh * hd)) + (headIdx * hd));
  bool dimValid = (lid < hd);
  if (dimValid) {
    unsigned int srcIdx = (srcBase + lid);
    unsigned int kDstIdx = (kCacheBase + lid);
    unsigned int vDstIdx = (vCacheBase + lid);
    __half kVal = newK[srcIdx];
    __half vVal = newV[srcIdx];
    kvCache[kDstIdx] = kVal;
    kvCache[vDstIdx] = vVal;
  }
}
