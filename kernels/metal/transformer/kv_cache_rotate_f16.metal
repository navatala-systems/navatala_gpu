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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_transformer_kv_cache_rotate_f16(device const uint* blockTables [[buffer(0)]], device const uint* contextLens [[buffer(1)]], device const uint* shiftAmount [[buffer(2)]], device const uint* numHeads [[buffer(3)]], device const uint* headDim [[buffer(4)]], device const uint* pageSize [[buffer(5)]], device const uint* maxBlocks [[buffer(6)]], device half* kvCache [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint groupId = ((uint)(int(__tgid.x)));
  uint nh = numHeads[0u];
  uint hd = headDim[0u];
  uint ps = pageSize[0u];
  uint mb = maxBlocks[0u];
  uint shift = shiftAmount[0u];
  uint batchIdx = (groupId / nh);
  uint headIdx = (groupId % nh);
  uint ctxLen = contextLens[batchIdx];
  bool posValid = (lid < ctxLen);
  uint srcPos = (lid + shift);
  bool srcValid = (srcPos < ctxLen);
  bool valid = (posValid && srcValid);
  uint blockTableBase = (batchIdx * mb);
  uint kvBlockStride = (2u * (nh * (ps * hd)));
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
