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
extern "C" __global__ void navatala_transformer_rope_interleaved_f32(const float* _input, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, const float* base, float* _output) {
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
    unsigned int dimIdx = (remainder % halfHd);
    unsigned int posIdx = ((batchIdx * sl) + seqIdx);
    unsigned int pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimIdxF32 = ((float)(dimIdx));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimIdxF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int idx0 = (baseOffset + (seqOffset + dimIdx));
    unsigned int idx1 = (idx0 + halfHd);
    float x0 = _input[idx0];
    float x1 = _input[idx1];
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1 = (x0Sin + x1Cos);
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}
