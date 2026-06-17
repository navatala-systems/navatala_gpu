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
__kernel void navatala_transformer_rope_backward_f16(__global const half* gradOutput, __global const uint* positions, __global const uint* batchSize, __global const uint* seqLen, __global const uint* headDim, __global const float* base, __global half* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  float ropeBase = base[(uint)(0u)];
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
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * (uint)(2u));
    uint dim1 = (dim0 + (uint)(1u));
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
    half dy0F16 = gradOutput[idx0];
    half dy1F16 = gradOutput[idx1];
    float dy0 = ((float)(dy0F16));
    float dy1 = ((float)(dy1F16));
    float dy0Cos = (dy0 * cosTheta);
    float dy1Sin = (dy1 * sinTheta);
    float dx0F32 = (dy0Cos + dy1Sin);
    float dy0Sin = (dy0 * sinTheta);
    float dy1Cos = (dy1 * cosTheta);
    float negDy0Sin = (as_float(0x00000000u) - dy0Sin);
    float dx1F32 = (negDy0Sin + dy1Cos);
    half dx0 = ((half)(dx0F32));
    half dx1 = ((half)(dx1F32));
    gradInput[idx0] = dx0;
    gradInput[idx1] = dx1;
  }
}
