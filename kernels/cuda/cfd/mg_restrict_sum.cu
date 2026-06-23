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
extern "C" __global__ void navatala_cfd_mg_restrict_sum(const float* rFine, const int* prolongRowOffsets, const int* prolongCols, const float* prolongVals, float* rCoarse, const int* mgCounts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) >= ((int)(mgCounts[0]))) {
    return;
  } else {
    int beg = prolongRowOffsets[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int c1 = ((int)(blockIdx.x * blockDim.x + threadIdx.x) + 1);
    int end = prolongRowOffsets[c1];
    float v = rFine[(int)(blockIdx.x * blockDim.x + threadIdx.x)];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int col = prolongCols[k];
      atomicAdd(&rCoarse[col], (prolongVals[k] * v));
    }
  }
}
