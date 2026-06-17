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

__kernel void navatala_transformer_pack_sequences_f32(__global const float* _input, __global const uint* seqLengths, __global const uint* seqOffsets, __global const uint* batchSize, __global const uint* maxSeqLen, __global const uint* hiddenSize, __global const uint* totalTokens, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint msl = maxSeqLen[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  uint tt = totalTokens[(uint)(0u)];
  uint totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint tokenIdx = (gid / hs);
    uint hiddenIdx = (gid % hs);
    uint foundBatchIdx = (uint)(0u);
    for (int b = 0; b < (int)(bs); ++b) {
      uint currOffset = seqOffsets[b];
      uint nextB = (b + (uint)(1u));
      uint nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    uint batchIdx = foundBatchIdx;
    uint batchOffset = seqOffsets[batchIdx];
    uint localTokenIdx = (tokenIdx - batchOffset);
    uint batchStride = (msl * hs);
    uint inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    float val = _input[inputIdx];
    _output[gid] = val;
  }
}
