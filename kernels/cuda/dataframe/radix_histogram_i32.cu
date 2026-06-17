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
extern "C" __global__ void navatala_dataframe_radix_histogram_i32(const int* _input, const unsigned int* count, const unsigned int* bitOffset, unsigned int* histogram) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  extern __shared__ unsigned int localHist[]; /* dynamic shared memory (best-effort) */
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int bits = bitOffset[0u];
  localHist[lid] = 0u;
  __syncthreads();
  bool inBounds = (gid < n);
  if (inBounds) {
    int val = _input[gid];
    unsigned int valU = ((unsigned int)(val));
    unsigned int signFlipped = (valU ^ 2147483648u);
    unsigned int digit = ((signFlipped >> bits) & 255u);
    atomicAdd(&localHist[digit], 1u);
  }
  __syncthreads();
  unsigned int localCount = localHist[lid];
  bool hasCount = (localCount > 0u);
  if (hasCount) {
    atomicAdd(&histogram[lid], localCount);
  }
}
