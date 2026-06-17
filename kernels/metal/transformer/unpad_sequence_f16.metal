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

kernel void navatala_transformer_unpad_sequence_f16(device const half* _input [[buffer(0)]], device const uint* seqLengths [[buffer(1)]], device const uint* seqOffsets [[buffer(2)]], device const uint* batchSize [[buffer(3)]], device const uint* maxSeqLen [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device const uint* totalTokens [[buffer(6)]], device half* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint msl = maxSeqLen[0u];
  uint hs = hiddenSize[0u];
  uint tt = totalTokens[0u];
  uint totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint tokenIdx = (gid / hs);
    uint hiddenIdx = (gid % hs);
    uint foundBatchIdx = 0u;
    for (int b = 0; b < (int)(bs); ++b) {
      uint currOffset = seqOffsets[b];
      uint nextB = (b + 1u);
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
    half val = _input[inputIdx];
    _output[gid] = val;
  }
}
