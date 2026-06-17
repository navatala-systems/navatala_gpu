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
__kernel void navatala_transformer_kv_cache_rotate_f16(__global const uint* blockTables, __global const uint* contextLens, __global const uint* shiftAmount, __global const uint* numHeads, __global const uint* headDim, __global const uint* pageSize, __global const uint* maxBlocks, __global half* kvCache) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint ps = pageSize[(uint)(0u)];
  uint mb = maxBlocks[(uint)(0u)];
  uint shift = shiftAmount[(uint)(0u)];
  uint batchIdx = (groupId / nh);
  uint headIdx = (groupId % nh);
  uint ctxLen = contextLens[batchIdx];
  bool posValid = (lid < ctxLen);
  uint srcPos = (lid + shift);
  bool srcValid = (srcPos < ctxLen);
  bool valid = (posValid && srcValid);
  uint blockTableBase = (batchIdx * mb);
  uint kvBlockStride = ((uint)(2u) * (nh * (ps * hd)));
  uint kvSlotStride = (nh * (ps * hd));
  uint kvHeadStride = (ps * hd);
  if (valid) {
    uint srcBlockIdx = (srcPos / ps);
    uint srcBlockOffset = (srcPos % ps);
    uint srcPhysBlock = blockTables[(blockTableBase + srcBlockIdx)];
    uint dstBlockIdx = (lid / ps);
    uint dstBlockOffset = (lid % ps);
    uint dstPhysBlock = blockTables[(blockTableBase + dstBlockIdx)];
    uint kSrcBase = ((srcPhysBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (srcBlockOffset * hd)));
    uint kDstBase = ((dstPhysBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (dstBlockOffset * hd)));
    uint vSrcBase = ((srcPhysBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (srcBlockOffset * hd))));
    uint vDstBase = ((dstPhysBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (dstBlockOffset * hd))));
    for (int d = 0; d < (int)(hd); ++d) {
      uint kSrcIdx = (kSrcBase + ((uint)(d)));
      uint kDstIdx = (kDstBase + ((uint)(d)));
      uint vSrcIdx = (vSrcBase + ((uint)(d)));
      uint vDstIdx = (vDstBase + ((uint)(d)));
      half kVal = kvCache[kSrcIdx];
      half vVal = kvCache[vSrcIdx];
      kvCache[kDstIdx] = kVal;
      kvCache[vDstIdx] = vVal;
    }
  }
}
