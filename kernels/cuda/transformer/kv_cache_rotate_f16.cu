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
extern "C" __global__ void navatala_transformer_kv_cache_rotate_f16(const unsigned int* blockTables, const unsigned int* contextLens, const unsigned int* shiftAmount, const unsigned int* numHeads, const unsigned int* headDim, const unsigned int* pageSize, const unsigned int* maxBlocks, __half* kvCache) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int ps = pageSize[0u];
  unsigned int mb = maxBlocks[0u];
  unsigned int shift = shiftAmount[0u];
  unsigned int batchIdx = (groupId / nh);
  unsigned int headIdx = (groupId % nh);
  unsigned int ctxLen = contextLens[batchIdx];
  bool posValid = (lid < ctxLen);
  unsigned int srcPos = (lid + shift);
  bool srcValid = (srcPos < ctxLen);
  bool valid = (posValid && srcValid);
  unsigned int blockTableBase = (batchIdx * mb);
  unsigned int kvBlockStride = (2u * (nh * (ps * hd)));
  unsigned int kvSlotStride = (nh * (ps * hd));
  unsigned int kvHeadStride = (ps * hd);
  if (valid) {
    unsigned int srcBlockIdx = (srcPos / ps);
    unsigned int srcBlockOffset = (srcPos % ps);
    unsigned int srcPhysBlock = blockTables[(blockTableBase + srcBlockIdx)];
    unsigned int dstBlockIdx = (lid / ps);
    unsigned int dstBlockOffset = (lid % ps);
    unsigned int dstPhysBlock = blockTables[(blockTableBase + dstBlockIdx)];
    unsigned int kSrcBase = ((srcPhysBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (srcBlockOffset * hd)));
    unsigned int kDstBase = ((dstPhysBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (dstBlockOffset * hd)));
    unsigned int vSrcBase = ((srcPhysBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (srcBlockOffset * hd))));
    unsigned int vDstBase = ((dstPhysBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (dstBlockOffset * hd))));
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int kSrcIdx = (kSrcBase + ((unsigned int)(d)));
      unsigned int kDstIdx = (kDstBase + ((unsigned int)(d)));
      unsigned int vSrcIdx = (vSrcBase + ((unsigned int)(d)));
      unsigned int vDstIdx = (vDstBase + ((unsigned int)(d)));
      __half kVal = kvCache[kSrcIdx];
      __half vVal = kvCache[vSrcIdx];
      kvCache[kDstIdx] = kVal;
      kvCache[vDstIdx] = vVal;
    }
  }
}
