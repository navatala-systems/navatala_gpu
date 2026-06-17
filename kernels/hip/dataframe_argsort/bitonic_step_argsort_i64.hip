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
extern "C" __global__ void navatala_dataframe_argsort_bitonic_step_argsort_i64(const long long* keys, const unsigned int* count, const unsigned int* stepSize, const unsigned int* halfBlockSize, unsigned int* indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int step = stepSize[0u];
  unsigned int halfBlock = halfBlockSize[0u];
  unsigned int pairIdx = (gid * 2u);
  unsigned int blockIdx = (pairIdx / step);
  unsigned int inBlock = (pairIdx % step);
  unsigned int i = ((blockIdx * step) + inBlock);
  unsigned int j = (i + halfBlock);
  bool inBounds = (j < n);
  if (inBounds) {
    unsigned int idxI = indices[i];
    unsigned int idxJ = indices[j];
    long long keyI = keys[idxI];
    long long keyJ = keys[idxJ];
    unsigned int blockPairIdx = (i / (step * 2u));
    bool ascending = ((blockPairIdx % 2u) == 0u);
    bool needSwapAsc = (keyI > keyJ);
    bool needSwapDesc = (keyI < keyJ);
    bool needSwap = ((ascending) ? (needSwapAsc) : (needSwapDesc));
    unsigned int newIdxI = ((needSwap) ? (idxJ) : (idxI));
    unsigned int newIdxJ = ((needSwap) ? (idxI) : (idxJ));
    indices[i] = newIdxI;
    indices[j] = newIdxJ;
  }
}
