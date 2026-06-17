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
#include <cuda_fp16.h>
extern "C" __global__ void navatala_transformer_unpad_sequence_f16(const __half* _input, const unsigned int* seqLengths, const unsigned int* seqOffsets, const unsigned int* batchSize, const unsigned int* maxSeqLen, const unsigned int* hiddenSize, const unsigned int* totalTokens, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int msl = maxSeqLen[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int tt = totalTokens[0u];
  unsigned int totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    unsigned int tokenIdx = (gid / hs);
    unsigned int hiddenIdx = (gid % hs);
    unsigned int foundBatchIdx = 0u;
    for (int b = 0; b < (int)(bs); ++b) {
      unsigned int currOffset = seqOffsets[b];
      unsigned int nextB = (b + 1u);
      unsigned int nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    unsigned int batchIdx = foundBatchIdx;
    unsigned int batchOffset = seqOffsets[batchIdx];
    unsigned int localTokenIdx = (tokenIdx - batchOffset);
    unsigned int batchStride = (msl * hs);
    unsigned int inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    __half val = _input[inputIdx];
    _output[gid] = val;
  }
}
