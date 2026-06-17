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
extern "C" __global__ void navatala_dataframe_argsort_bitonic_step_sort_by_key_i32(const unsigned int* count, const unsigned int* stepSize, const unsigned int* halfBlockSize, int* keys, int* values) {
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
    int keyI = keys[i];
    int keyJ = keys[j];
    int valI = values[i];
    int valJ = values[j];
    unsigned int blockPairIdx = (i / (step * 2u));
    bool ascending = ((blockPairIdx % 2u) == 0u);
    bool needSwapAsc = (keyI > keyJ);
    bool needSwapDesc = (keyI < keyJ);
    bool needSwap = ((ascending) ? (needSwapAsc) : (needSwapDesc));
    int newKeyI = ((needSwap) ? (keyJ) : (keyI));
    int newKeyJ = ((needSwap) ? (keyI) : (keyJ));
    int newValI = ((needSwap) ? (valJ) : (valI));
    int newValJ = ((needSwap) ? (valI) : (valJ));
    keys[i] = newKeyI;
    keys[j] = newKeyJ;
    values[i] = newValI;
    values[j] = newValJ;
  }
}
