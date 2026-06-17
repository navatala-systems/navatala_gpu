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
extern "C" __global__ void navatala_transformer_precompute_rotary_emb_f32(const unsigned int* maxSeqLen, const unsigned int* headDim, const float* base, float* cosOutput, float* sinOutput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int msl = maxSeqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int total = (msl * halfHd);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int pos = (gid / halfHd);
    unsigned int dimPair = (gid % halfHd);
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosVal = cos(theta);
    float sinVal = sin(theta);
    cosOutput[gid] = cosVal;
    sinOutput[gid] = sinVal;
  }
}
