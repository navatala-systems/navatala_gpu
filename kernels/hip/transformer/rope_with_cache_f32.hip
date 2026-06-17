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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_rope_with_cache_f32(const float* _input, const float* cosCache, const float* sinCache, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int hd = headDim[0u];
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
    unsigned int cacheIdx = ((pos * halfHd) + dimPair);
    float cosTheta = cosCache[cacheIdx];
    float sinTheta = sinCache[cacheIdx];
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int dim0 = (dimPair * 2u);
    unsigned int dim1 = (dim0 + 1u);
    unsigned int idx0 = (baseOffset + (seqOffset + dim0));
    unsigned int idx1 = (baseOffset + (seqOffset + dim1));
    float x0 = _input[idx0];
    float x1 = _input[idx1];
    float y0 = ((x0 * cosTheta) - (x1 * sinTheta));
    float y1 = ((x0 * sinTheta) + (x1 * cosTheta));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}
