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
extern "C" __global__ void navatala_transformer_rope_backward_f32(const float* gradOutput, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, const float* base, float* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int batchSeqPairs = (sl * halfHd);
    unsigned int batchIdx = (gid / batchSeqPairs);
    unsigned int remainder = (gid % batchSeqPairs);
    unsigned int seqIdx = (remainder / halfHd);
    unsigned int dimPair = (remainder % halfHd);
    unsigned int posIdx = ((batchIdx * sl) + seqIdx);
    unsigned int pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int dim0 = (dimPair * 2u);
    unsigned int dim1 = (dim0 + 1u);
    unsigned int idx0 = (baseOffset + (seqOffset + dim0));
    unsigned int idx1 = (baseOffset + (seqOffset + dim1));
    float dy0 = gradOutput[idx0];
    float dy1 = gradOutput[idx1];
    float dy0Cos = (dy0 * cosTheta);
    float dy1Sin = (dy1 * sinTheta);
    float dx0 = (dy0Cos + dy1Sin);
    float dy0Sin = (dy0 * sinTheta);
    float dy1Cos = (dy1 * cosTheta);
    float negDy0Sin = (__uint_as_float(0x00000000u) - dy0Sin);
    float dx1 = (negDy0Sin + dy1Cos);
    gradInput[idx0] = dx0;
    gradInput[idx1] = dx1;
  }
}
