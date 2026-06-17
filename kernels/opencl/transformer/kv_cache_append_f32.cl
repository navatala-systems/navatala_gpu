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

__kernel void navatala_transformer_kv_cache_append_f32(__global const float* newK, __global const float* newV, __global const uint* blockTables, __global const uint* seqPositions, __global const uint* numHeads, __global const uint* headDim, __global const uint* pageSize, __global const uint* maxBlocks, __global float* kvCache) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint ps = pageSize[(uint)(0u)];
  uint mb = maxBlocks[(uint)(0u)];
  uint batchIdx = (groupId / nh);
  uint headIdx = (groupId % nh);
  uint seqPos = seqPositions[batchIdx];
  uint blockIdx = (seqPos / ps);
  uint blockOffset = (seqPos % ps);
  uint blockTableBase = (batchIdx * mb);
  uint physicalBlock = blockTables[(blockTableBase + blockIdx)];
  uint kvBlockStride = ((uint)(2u) * (nh * (ps * hd)));
  uint kvSlotStride = (nh * (ps * hd));
  uint kvHeadStride = (ps * hd);
  uint kCacheBase = ((physicalBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (blockOffset * hd)));
  uint vCacheBase = ((physicalBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (blockOffset * hd))));
  uint srcBase = ((batchIdx * (nh * hd)) + (headIdx * hd));
  bool dimValid = (lid < hd);
  if (dimValid) {
    uint srcIdx = (srcBase + lid);
    uint kDstIdx = (kCacheBase + lid);
    uint vDstIdx = (vCacheBase + lid);
    float kVal = newK[srcIdx];
    float vVal = newV[srcIdx];
    kvCache[kDstIdx] = kVal;
    kvCache[vDstIdx] = vVal;
  }
}
