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

kernel void navatala_transformer_rope_backward_f32(device const float* gradOutput [[buffer(0)]], device const uint* positions [[buffer(1)]], device const uint* batchSize [[buffer(2)]], device const uint* seqLen [[buffer(3)]], device const uint* headDim [[buffer(4)]], device const float* base [[buffer(5)]], device float* gradInput [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint bs = batchSize[0u];
  uint sl = seqLen[0u];
  uint hd = headDim[0u];
  float ropeBase = base[0u];
  uint halfHd = (hd / 2u);
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
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_type<float>(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * 2u);
    uint dim1 = (dim0 + 1u);
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
    float dy0 = gradOutput[idx0];
    float dy1 = gradOutput[idx1];
    float dy0Cos = (dy0 * cosTheta);
    float dy1Sin = (dy1 * sinTheta);
    float dx0 = (dy0Cos + dy1Sin);
    float dy0Sin = (dy0 * sinTheta);
    float dy1Cos = (dy1 * cosTheta);
    float negDy0Sin = (as_type<float>(0x00000000u) - dy0Sin);
    float dx1 = (negDy0Sin + dy1Cos);
    gradInput[idx0] = dx0;
    gradInput[idx1] = dx1;
  }
}
