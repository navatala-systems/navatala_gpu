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

__kernel void navatala_transformer_rope_with_cache_f32(__global const float* _input, __global const float* cosCache, __global const float* sinCache, __global const uint* positions, __global const uint* batchSize, __global const uint* seqLen, __global const uint* headDim, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint halfHd = (hd / (uint)(2u));
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimPair = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    uint cacheIdx = ((pos * halfHd) + dimPair);
    float cosTheta = cosCache[cacheIdx];
    float sinTheta = sinCache[cacheIdx];
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * (uint)(2u));
    uint dim1 = (dim0 + (uint)(1u));
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
    float x0 = _input[idx0];
    float x1 = _input[idx1];
    float y0 = ((x0 * cosTheta) - (x1 * sinTheta));
    float y1 = ((x0 * sinTheta) + (x1 * cosTheta));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}
