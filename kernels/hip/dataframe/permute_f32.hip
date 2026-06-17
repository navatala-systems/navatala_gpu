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
extern "C" __global__ void navatala_dataframe_permute_f32(float* data, const unsigned int* random, const unsigned int* n) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0u];
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    for (int i = 0; i < (int)(nVal); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int idx = ((nVal - 1u) - iU32);
      bool shouldSwap = (idx > 0u);
      if (shouldSwap) {
        unsigned int randIdx = ((nVal - 1u) - idx);
        unsigned int randVal = random[randIdx];
        unsigned int idxPlusOne = (idx + 1u);
        unsigned int j = (randVal % idxPlusOne);
        float valIdx = data[idx];
        float valJ = data[j];
        data[idx] = valJ;
        data[j] = valIdx;
      }
    }
  }
}
