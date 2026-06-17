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
extern "C" __global__ void navatala_transformer_pad_sequence_f32(const float* _input, const unsigned int* seqLengths, const unsigned int* seqOffsets, const unsigned int* batchSize, const unsigned int* maxSeqLen, const unsigned int* hiddenSize, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int msl = maxSeqLen[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    unsigned int seqHidden = (msl * hs);
    unsigned int batchIdx = (gid / seqHidden);
    unsigned int remainder = (gid % seqHidden);
    unsigned int seqIdx = (remainder / hs);
    unsigned int hiddenIdx = (remainder % hs);
    unsigned int seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      unsigned int seqOffset = seqOffsets[batchIdx];
      unsigned int inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      float val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = __uint_as_float(0x00000000u);
    }
  }
}
