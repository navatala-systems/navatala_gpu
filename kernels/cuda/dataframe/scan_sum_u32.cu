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
extern "C" __global__ void navatala_dataframe_scan_sum_u32(const unsigned int* _input, const unsigned int* count, const unsigned int* scanMode, unsigned int* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ unsigned int sdata[512];
  unsigned int n = count[0u];
  unsigned int mode = scanMode[0u];
  bool isExclusive = (mode == 1u);
  unsigned int ai = lid;
  unsigned int bi = (lid + 256u);
  bool inBoundsA = (ai < n);
  bool inBoundsB = (bi < n);
  unsigned int valA = ((inBoundsA) ? (_input[ai]) : (0u));
  unsigned int valB = ((inBoundsB) ? (_input[bi]) : (0u));
  sdata[ai] = valA;
  sdata[bi] = valB;
  __syncthreads();
  for (int offset = 0; offset < (int)(1); ++offset) {
    unsigned int offsetU32 = ((unsigned int)(offset));
    unsigned int upMask = ((offsetU32 * 2u) - 1u);
    bool upShouldWork = ((lid & upMask) == 0u);
    if (upShouldWork) {
      unsigned int upAi2 = ((offsetU32 * ((lid * 2u) + 1u)) + (offsetU32 - 1u));
      unsigned int upBi2 = (upAi2 + offsetU32);
      unsigned int upVa = sdata[upAi2];
      unsigned int upVb = sdata[upBi2];
      sdata[upBi2] = (upVa + upVb);
    }
    __syncthreads();
  }
  if ((lid == 0u)) {
    sdata[511u] = 0u;
  }
  __syncthreads();
  for (int offset = 0; offset < (int)(256); ++offset) {
    unsigned int downOffsetU32 = ((unsigned int)(offset));
    unsigned int downMask = ((downOffsetU32 * 2u) - 1u);
    bool downShouldWork = ((lid & downMask) == 0u);
    if (downShouldWork) {
      unsigned int downAi2 = ((downOffsetU32 * ((lid * 2u) + 1u)) + (downOffsetU32 - 1u));
      unsigned int downBi2 = (downAi2 + downOffsetU32);
      unsigned int downT = sdata[downAi2];
      unsigned int downVb = sdata[downBi2];
      sdata[downAi2] = downVb;
      sdata[downBi2] = (downT + downVb);
    }
    __syncthreads();
  }
  if (inBoundsA) {
    unsigned int exclusiveResultA = sdata[ai];
    unsigned int inclusiveResultA = (exclusiveResultA + valA);
    unsigned int resultA = ((isExclusive) ? (exclusiveResultA) : (inclusiveResultA));
    _output[ai] = resultA;
  }
  if (inBoundsB) {
    unsigned int exclusiveResultB = sdata[bi];
    unsigned int inclusiveResultB = (exclusiveResultB + valB);
    unsigned int resultB = ((isExclusive) ? (exclusiveResultB) : (inclusiveResultB));
    _output[bi] = resultB;
  }
}
